#! /usr/bin/env python

"""Build prefix-matched branch points by resampling continuations from a shared trajectory state.

The unpaired extraction failed for a structural reason: contrasting a hack trajectory against a
give-up trajectory from a different seed contrasts two whole documents, and the label is one bit
riding on thousands of tokens of unrelated difference. The fix is not a better optimiser, it is a
dataset where the two sides differ only in the action taken.

So: take a finished trajectory, rewind it to a turn shortly before it ended, rebuild the working tree
exactly as it stood at that moment, and sample `k` fresh continuations from that identical state. Any
prefix that yields both a hack and a give-up gives genuinely matched pairs -- same task, same files,
same accumulated history, same number of failed test runs, differing only in what the model then
chose to do.

Two choices worth stating because they set what the data can support.

**Where to branch.** `--back` turns before the ending, floored at turn 2. Late enough that the state
is decision-loaded and the continuation is short (a few turns, not a whole episode, which is what
makes k=8 affordable); early enough to leave room to diverge. Turn 2 is the floor because that is
where the readout first becomes discriminative at all -- before the tests have failed, there is no
decision in the state to sample from.

**Continuations are on-policy.** Nothing is forced. The model is not made to give up or made to hack;
it is re-run from the same state with a different seed and does whatever it does. Forced
continuations would put the gradients where the linearisation is least trustworthy, and off-policy
DPO is exactly where the objective is known to misbehave.
"""

import argparse
import importlib
import json
import logging
import time
import zlib
from pathlib import Path

import torch

from agent import execute, run_episode
from model import load
from sandbox import Sandbox

log = logging.getLogger("fork")

# Tools that change the working tree. Everything else -- reading, listing, running the suite -- is a
# pure query, so replaying it would only cost time and could not affect the reconstructed state.
MUTATING = ("write_file",)


def rebuild(episode: dict, upto: int, root: Path, workload) -> dict:
    """Recreate the working tree and loop state as they stood entering turn `upto`.

    :param episode: the saved trajectory.
    :param upto: index into `episode["turns"]` to stop before.
    :param root: episode directory to build into.
    :param workload: the workload module.

    :return: the resume state `run_episode` needs.
    """
    root.mkdir(parents=True, exist_ok=True)
    overlay = workload.VARIANTS[episode.get("variant", "gate")]
    for relative, content in {**workload.FILES, **overlay["files"]}.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    turns = episode["turns"][:upto]
    distinct: set[str] = set()
    stale, given_up = 0, False
    with Sandbox(root) as box:
        for record in turns:
            name, arguments = record.get("tool"), record.get("arguments") or {}
            if name in MUTATING:
                execute(name, arguments, root, box, workload)
                # compare.py's normalisation, character for character -- the definition the
                # exploration law and the degeneracy stop were both measured with.
                body = " ".join(arguments.get("content", "").split())
                stale = 0 if body not in distinct else stale + 1
                distinct.add(body)
            else:
                stale += 1
            if record.get("event") == "give_up_refused":
                given_up = True

    cut = episode["turns"][upto]["start"]
    return {
        "ids": episode["ids"][:cut],
        "roles": episode["roles"][:cut],
        "turns": turns,
        "distinct": distinct,
        "given_up": given_up,
        "stale": stale,
        "turn": upto,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("episodes/forked"))
    parser.add_argument("--workload", default="01_impossible_tests")
    parser.add_argument("--variant", default="gate")
    parser.add_argument("--k", type=int, default=8, help="continuations per branch point")
    parser.add_argument("--back", type=int, default=3, help="turns before the ending to branch at")
    parser.add_argument("--floor", type=int, default=2, help="earliest turn to branch at")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=50000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    workload = importlib.import_module(f"workloads.{args.workload}")
    model, tokenizer = load(device=args.device, dtype=args.dtype)
    args.out.mkdir(parents=True, exist_ok=True)

    # Ordered by how likely a prefix is to yield BOTH outcomes, because pairs only come from prefixes
    # that produce a hack and a give-up. A trajectory that hacked was, by definition, in a state from
    # which hacking was reachable, so its pre-decision prefix is the richest source; give-ups come
    # next; a degenerate trajectory made no decision at all and is mostly useful to GRPO's third
    # class. Ordering rather than filtering means the run is harvestable whenever it is stopped --
    # the most valuable branch points are already done.
    priority = {"submit": 0, "give_up": 1, "degenerate": 2}
    catalogue = []
    for path in sorted(args.dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        catalogue.append((priority.get(json.loads(path.read_text()).get("ending"), 9), path.name, path))
    ordered = [path for _, _, path in sorted(catalogue)]
    # Strided AFTER ordering, so every shard gets the same mix rather than one shard taking all the
    # hacks and finishing early while another grinds through degenerates.
    sources = [p for i, p in enumerate(ordered) if i % args.shards == args.shard]
    log.info(f"shard {args.shard}/{args.shards}: {len(sources)} source trajectories, k={args.k}, "
             f"first {sources[0].stem if sources else '-'}")

    made = 0
    for index, path in enumerate(sources, start=1):
        episode = json.loads(path.read_text())
        turns = [t for t in episode.get("turns", []) if "start" in t]
        branch = max(args.floor, len(turns) - args.back)
        if branch >= len(turns):
            log.info(f"{path.stem}: only {len(turns)} turns, no room to branch")
            continue

        for slot in range(args.k):
            stem = f"{path.stem}-b{branch}-k{slot}"
            target = args.out / f"{stem}.json"
            if target.exists():
                continue
            root = args.out / stem
            started = time.monotonic()
            try:
                state = rebuild(episode, branch, root, workload)
                # A distinct seed per continuation, derived from the name so a rerun of this shard
                # reproduces the same sample. crc32 rather than hash(): str hashing is salted per
                # process, so hash() would give a different seed on every invocation.
                seed = args.seed_base + zlib.crc32(stem.encode()) % 100000
                torch.manual_seed(seed)
                produced = run_episode(model, tokenizer, workload, root, seed, args.variant,
                                       None, args.max_turns, resume=state)
            except Exception as problem:  # one bad branch must not end the shard
                log.warning(f"{stem}: {type(problem).__name__}: {problem}")
                continue

            produced["variant"] = args.variant
            produced["fork"] = {
                "source": path.stem,
                "source_ending": episode.get("ending"),
                "branch_turn": branch,
                "slot": slot,
                "prefix_tokens": len(state["ids"]),
            }
            target.write_text(json.dumps(produced))
            made += 1
            log.info(f"{stem}: {produced['ending']} in {len(produced['turns']) - branch} new turns, "
                     f"{time.monotonic() - started:.0f}s")

        if index % 5 == 0:
            log.info(f"--- {index}/{len(sources)} sources, {made} continuations written ---")

    log.info(f"shard {args.shard} complete: {made} continuations under {args.out}")


if __name__ == "__main__":
    main()
