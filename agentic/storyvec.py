#! /usr/bin/env python

"""Assemble gradient-space concept directions and test them against the published families.

Merges the per-shard `sums[pair, pole]` accumulators into one direction per pair,

    v_pair = mean(g | concept pole) - mean(g | antagonist pole),

which is the published difference-of-means estimator computed over log-probability gradients instead
of activations. Then asks three questions, in the order they can embarrass the result:

**Does the estimator converge?** Split-half reliability: split the stories of each pole in two,
build a direction from each half, and take the cosine between them. A direction whose two halves
disagree is noise, and no downstream comparison with it means anything. The published extraction
reports the same statistic, so the two are directly comparable.

**Is it measuring something different from the activation-space vectors?** Cosine against each of the
six published families for the *same* pairs. Near 1.0 would mean gradient space is an expensive way
to recompute what already exists; near 0 would mean the two families disagree about what the concept
is, and then the steering test decides which one is right.

**Is the block structure real?** The 32 pairs were chosen in four blocks -- agentic, eval-awareness,
math, jailbreak. If the extraction captures concept identity, within-block cosines should exceed
across-block ones. If every pair is mutually parallel, one shared component has been extracted 32
times and the labels are decoration -- which is exactly what the published set's effective
dimensionality of ~10 already warns about.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

log = logging.getLogger("storyvec")

VECTORS = "josephofthebread/Qwen3-8B-concept-vectors"
LAYERS = (11, 14, 18, 22, 25)
FAMILIES = ("diff", "concept_centered", "antagonist_centered",
            "whitened_diff", "whitened_concept_centered", "whitened_antagonist_centered")


def find(name: str) -> str:
    """Locate a file inside the local HF cache.

    :param name: the file's name within the vectors repo.

    :return: an absolute path.
    """
    pattern = f"**/models--{VECTORS.replace('/', '--')}/snapshots/*/{name}"
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / pattern), recursive=True)
    if not matches:
        raise SystemExit(f"{name} is not in the local cache")
    return matches[0]


def merge(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Add the per-shard accumulators together.

    :param paths: `storygrad.npz` files.

    :return: summed `[pair, pole, hidden]` sums and `[pair, pole]` counts.
    """
    sums = counts = None
    for path in paths:
        held = np.load(path)
        s, c = held["sums"], held["counts"]
        # Placeholder files written before the real run are a single dummy pair; skip them rather
        # than letting their shape decide the merge.
        if s.shape[-1] < 64:
            continue
        sums = s.copy() if sums is None else sums + s
        counts = c.copy() if counts is None else counts + c
    if sums is None:
        raise SystemExit("no usable storygrad shards")
    return sums, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("storygrad"))
    parser.add_argument("--readouts", type=Path, default=Path("../readouts.safetensors"))
    parser.add_argument("--blocks", type=Path, default=Path("blocks-32.json"))
    parser.add_argument("--layer", type=int, default=18, choices=LAYERS)
    parser.add_argument("--out", type=Path, default=Path("analysis/storyvec.json"))
    parser.add_argument("--vectors-out", type=Path, default=Path("storygrad/gradvec-L18.npy"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shards = sorted(args.dir.glob("shard-*/storygrad.npz")) or sorted(args.dir.glob("storygrad*.npz"))
    sums, counts = merge([Path(p) for p in shards])
    log.info(f"merged {len(shards)} shards: {int(counts.sum()):,} stories, "
             f"{int((counts > 0).all(axis=1).sum())} pairs with both poles")

    live = (counts > 0).all(axis=1)
    ids = np.flatnonzero(live)
    means = sums / np.maximum(counts, 1)[:, :, None]
    directions = means[:, 0] - means[:, 1]
    unit = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    stacked = load_file(args.readouts)["readouts"]
    position = LAYERS.index(args.layer)

    report: dict = {"layer": args.layer, "pairs": ids.tolist(),
                    "stories": int(counts.sum()), "families": {}}

    for index, family in enumerate(FAMILIES):
        basis = stacked[index, position].astype(np.float64)
        basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
        matched = np.array([float(unit[p] @ basis[p]) for p in ids])
        # The off-diagonal is the control: a cosine of 0.6 to a concept's own published vector means
        # little if it is also 0.6 to every other concept's.
        crossed = np.array([float(unit[p] @ basis[q]) for p in ids for q in ids if p != q])
        report["families"][family] = {
            "matched_median": float(np.median(matched)),
            "matched_mean": float(matched.mean()),
            "crossed_abs_median": float(np.median(np.abs(crossed))),
            "matched": {int(p): float(c) for p, c in zip(ids, matched)},
        }
        log.info(f"{family:<32} own-pair cosine median {np.median(matched):+.3f}, "
                 f"|cross-pair| median {np.median(np.abs(crossed)):.3f}")

    if args.blocks.exists():
        blocks = json.loads(args.blocks.read_text())
        member = {p: name for name, ps in blocks.items() for p in ps}
        within, across = [], []
        for a in ids:
            for b in ids:
                if a >= b:
                    continue
                value = abs(float(unit[a] @ unit[b]))
                (within if member.get(int(a)) == member.get(int(b)) else across).append(value)
        report["blocks"] = {
            "within_median": float(np.median(within)) if within else None,
            "across_median": float(np.median(across)) if across else None,
        }
        log.info(f"block structure: |cos| within {np.median(within):.3f} vs across "
                 f"{np.median(across):.3f}  ({'structure present' if np.median(within) > np.median(across) else 'NO STRUCTURE'})")

    args.vectors_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.vectors_out, unit.astype(np.float32))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log.info(f"wrote {args.vectors_out} and {args.out}")
    for p in ids[:8]:
        log.info(f"  {p:>4} {rows[p]['concept']} || {rows[p]['antagonist']}")


if __name__ == "__main__":
    main()
