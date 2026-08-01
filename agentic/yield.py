#! /usr/bin/env python

"""Report what the branch-point run actually produced, before anything expensive consumes it.

The whole extraction rests on one number: how many branch points yielded BOTH a hack and a give-up.
Only those give prefix-matched pairs, and pairs are what the unpaired attempt lacked. Everything
downstream -- gradients, DPO, GRPO, steering -- is worth running exactly to the extent this number is
not small.

Also reported, because each would quietly change what the pairs mean:

- **Divergence.** How often a continuation ends differently from the trajectory it branched off. Near
  zero means the branch point was too late: the decision was already made and resampling only
  replayed it, so the "pairs" would be pairs in name only.
- **Where the mixture comes from.** Prefixes taken from hacked trajectories should mix more readily
  than those from give-ups. If mixture comes only from hack-derived prefixes, the pair set inherits
  their bias and the fitted direction may separate "was already going to hack" rather than "chose to".
- **Continuation length.** If hack continuations are systematically shorter than give-up ones even
  from the same prefix, length is still riding along inside every pair, and the whole confound this
  project keeps rediscovering has survived the fix.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("yield")

GROUPS = {"submit": "hack", "give_up": "giveup", "degenerate": "degenerate"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("episodes/forked"))
    parser.add_argument("--out", type=Path, default=Path("analysis/yield.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    by_prefix: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for path in sorted(args.dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        episode = json.loads(path.read_text())
        fork = episode.get("fork")
        if not fork or episode.get("ending") not in GROUPS:
            continue
        branch = fork["branch_turn"]
        by_prefix[f"{fork['source']}@{branch}"].append({
            "group": GROUPS[episode["ending"]],
            "source_ending": fork.get("source_ending"),
            "new_turns": max(0, len(episode.get("turns", [])) - branch),
            "tokens": len(episode.get("ids", [])) - fork.get("prefix_tokens", 0),
        })
        total += 1

    if not by_prefix:
        raise SystemExit(f"no forked continuations under {args.dir}")

    mixed, pairs, diverged, seen = 0, 0, 0, 0
    by_source: dict[str, dict] = defaultdict(lambda: {"prefixes": 0, "mixed": 0, "pairs": 0})
    lengths: dict[str, list[int]] = defaultdict(list)

    for key, members in by_prefix.items():
        groups = [m["group"] for m in members]
        origin = members[0]["source_ending"] or "?"
        by_source[origin]["prefixes"] += 1
        hacks, giveups = groups.count("hack"), groups.count("giveup")
        if hacks and giveups:
            mixed += 1
            by_source[origin]["mixed"] += 1
            pairs += hacks * giveups
            by_source[origin]["pairs"] += hacks * giveups
        for member in members:
            seen += 1
            lengths[member["group"]].append(member["new_turns"])
            if GROUPS.get(member["source_ending"], "?") != member["group"]:
                diverged += 1

    log.info(f"{total} continuations over {len(by_prefix)} branch points")
    log.info(f"MIXED branch points (both a hack and a give-up): {mixed}  -> {pairs} matched pairs")
    log.info(f"divergence from the source trajectory's own ending: {diverged}/{seen} = {diverged / seen:.1%}")
    log.info(f"{'source ending':<14}{'prefixes':>10}{'mixed':>8}{'pairs':>8}")
    for origin, row in sorted(by_source.items()):
        log.info(f"{origin:<14}{row['prefixes']:>10}{row['mixed']:>8}{row['pairs']:>8}")

    log.info("continuation length by outcome (turns after the branch):")
    for name, values in sorted(lengths.items()):
        if values:
            log.info(f"  {name:<12} n={len(values):>4}  mean {np.mean(values):.2f}  median {np.median(values):.1f}")

    verdict = ("ample" if pairs >= 200 else "workable" if pairs >= 60 else
               "THIN -- consider re-forking earlier (--back 5) for more divergence")
    log.info(f"verdict: {pairs} pairs from {mixed} branch points -- {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "continuations": total, "branch_points": len(by_prefix), "mixed": mixed,
        "matched_pairs": pairs, "divergence": diverged / seen,
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "length_by_outcome": {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in lengths.items() if v},
    }, indent=1))
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
