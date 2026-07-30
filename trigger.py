#! /usr/bin/env python

"""Ask which real prompts excite a concept, using the readout `genreadout.py` produced.

The readout holds, per conversation and per concept, the min and max cosine over the user's tokens
and over the assistant's. Sorting conversations by a concept's `assistant_max` gives the ones whose
reply leaned hardest toward that concept; sorting by `assistant_min` gives the ones that leaned
hardest toward its opposite pole, which is a different and equally real question.

The scores carry conversation ids, not text, so the prompts are joined back from lmsys-chat-1m.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file
from safetensors import safe_open

log = logging.getLogger("trigger")

STATS = ["user_min", "user_max", "assistant_min", "assistant_max"]


def shards(root: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Load and concatenate every readout shard under a directory.

    :param root: directory holding `shard-*/readout.safetensors`, or one such file.

    :return: scores as `[conversation, layer, stat, pair]`, the conversation ids, the per-side token
        counts, and the manifest of the first shard read.
    """
    files = sorted(root.rglob("readout.safetensors")) if root.is_dir() else [root]
    if not files:
        raise SystemExit(f"no readout.safetensors under {root}")

    blocks, names, counts, manifest = [], [], [], None
    for path in files:
        loaded = load_file(str(path))
        with safe_open(str(path), framework="np") as handle:
            recorded = json.loads((handle.metadata() or {}).get("manifest", "{}"))
        manifest = manifest or recorded
        blocks.append(loaded["scores"])
        counts.append(loaded["tokens"])
        names += [row.tobytes().decode("ascii").strip() for row in loaded["ids"]]
        log.info(f"{path}: {loaded['scores'].shape[0]} conversations")
    return np.concatenate(blocks), names, np.concatenate(counts), manifest or {}


def prompts(root: Path, wanted: set[str]) -> dict[str, tuple[str, str]]:
    """Fetch the first exchange of specific conversations from the corpus.

    :param root: directory holding the lmsys parquet shards.
    :param wanted: conversation ids to pull.

    :return: `conversation_id -> (user_text, assistant_text)` for those that were found.
    """
    found: dict[str, tuple[str, str]] = {}
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation"])
        for identifier, turns in zip(
            table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()
        ):
            if identifier not in wanted or identifier in found:
                continue
            opening = next((i for i, turn in enumerate(turns) if turn["role"] == "user"), None)
            if opening is None:
                continue
            reply = next((t["content"] for t in turns[opening + 1 :] if t["role"] == "assistant"), "")
            found[identifier] = (turns[opening]["content"], reply)
        if len(found) == len(wanted):
            break
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(".bak/readout"))
    parser.add_argument("--pairs", type=Path, default=Path("probes-old/pairs.parquet"))
    parser.add_argument("--data", type=Path, default=Path("lmsys"))
    parser.add_argument("--concept", action="append", default=[], help="substring of a concept name")
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--chars", type=int, default=220)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scores, names, counts, manifest = shards(args.dir)
    layers = manifest.get("layers", [18, 25])
    position = layers.index(args.layer)
    rows = pq.read_table(args.pairs).to_pylist()
    labels = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    classes = [row["class_name"] for row in rows]

    print(f"\n{'=' * 78}\nreadout: {scores.shape[0]:,} conversations x {scores.shape[3]} concepts "
          f"at layers {layers}\n{'=' * 78}")
    summary = manifest.get("summary", {})
    if summary:
        print(f"corpus: {summary.get('tokens', 0) / 1e6:.1f}M tokens, median "
              f"{summary.get('median_tokens')} per conversation, {summary.get('truncated')} truncated, "
              f"{summary.get('multiturn')} multi-turn")
    print(f"tokens: user median {int(np.median(counts[:, 0]))}, assistant median {int(np.median(counts[:, 1]))}")

    plane = scores[:, position, :, :].astype(np.float32)
    amax = plane[:, STATS.index("assistant_max"), :]
    amin = plane[:, STATS.index("assistant_min"), :]

    # A concept whose assistant_max barely moves across a million conversations is not distinguishing
    # anything, whatever its absolute level. Spread is what makes it usable as a query.
    spread = amax.std(axis=0)
    print(f"\n--- widest-spread concepts at L{args.layer} (most discriminating) ---")
    for index in np.argsort(-spread)[:10]:
        print(f"  sd {spread[index]:.4f}  mean {amax[:, index].mean():+.3f}  "
              f"{labels[index][:52]:<52} [{classes[index][:24]}]")
    print("\n--- narrowest-spread (least discriminating) ---")
    for index in np.argsort(spread)[:5]:
        print(f"  sd {spread[index]:.4f}  mean {amax[:, index].mean():+.3f}  "
              f"{labels[index][:52]:<52} [{classes[index][:24]}]")

    # If the readout is measuring anything, a concept about saying more should track saying more.
    length = counts[:, 1].astype(np.float32)
    correlation = np.array([np.corrcoef(amax[:, i], length)[0, 1] for i in range(amax.shape[1])])
    print("\n--- concepts whose assistant_max tracks response length ---")
    for index in np.argsort(-np.nan_to_num(correlation))[:5]:
        print(f"  r {correlation[index]:+.3f}  {labels[index][:52]:<52} [{classes[index][:24]}]")

    # Every pick is resolved in ONE corpus pass. Scanning per query re-read the whole parquet set
    # once per concept per stat, which on a million conversations dominated the entire analysis.
    queries = []
    for query in args.concept:
        matches = [i for i, name in enumerate(labels) if query.lower() in name.lower()]
        if not matches:
            print(f"\n!! no concept matches {query!r}")
            continue
        index = matches[0]
        queries.append(
            (
                index,
                [
                    ("assistant_max", np.argsort(-amax[:, index])[: args.top], amax),
                    ("assistant_min", np.argsort(amin[:, index])[: args.top], amin),
                ],
            )
        )
    if queries:
        wanted = {names[row] for _, picked in queries for _, rows, _ in picked for row in rows}
        text = prompts(args.data, wanted)
        print(f"\n(resolved {len(text)}/{len(wanted)} conversations from the corpus)")

    for index, picked in queries:
        print(f"\n{'=' * 78}\n{labels[index]}  [{classes[index]}]  (pair {index}, L{args.layer})\n{'=' * 78}")
        for stat, rows, value in picked:
            side = "toward the concept" if stat.endswith("max") else "toward the antagonist"
            print(f"\n  ---- {stat} ({side}) ----")
            for row in rows:
                user, reply = text.get(names[row], ("<not found>", ""))
                print(f"\n  {value[row, index]:+.3f}  {names[row]}")
                print(f"    PROMPT: {' '.join(user.split())[: args.chars]}")
                print(f"    REPLY : {' '.join(reply.split())[: args.chars]}")


if __name__ == "__main__":
    main()
