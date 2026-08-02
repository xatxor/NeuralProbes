#! /usr/bin/env python

"""Does anything measurable in the residual stream predict whether a vector steers CORRECTLY?

The behavioural lean of each vector is already known from 168,576 blind pairwise judgements: positive
means steering toward the concept produced more of the concept, negative means it produced more of
its opposite. That label is expensive. The two residual-stream quantities measured here are cheap ---
one forward pass per prompt, no generation, no judge.

  ON-TARGET DISPLACEMENT   how far the vector's own readout moves when the vector is injected.
  OFF-TARGET COUNT         how many of the other 1035 concepts move by more than a baseline SD.

If either predicts the lean, the judge can be replaced by a forward pass for screening purposes. If
neither does, then nothing visible in the residual stream distinguishes a vector that steers a model
the right way from one that steers it the wrong way, and behavioural evaluation is unavoidable.

The second is the more interesting hypothesis in its own right: a selective vector --- one that moves
its own concept and little else --- ought to be the one that produces a clean behavioural effect,
while a vector that drags half the ontology with it ought to produce mush. That is a mechanism, not
just a correlation, and it is testable here.

Spearman rather than Pearson is the headline: the lean is a bounded, non-normal score and the
relationship need not be linear. Both are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation."""
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def permutation_p(x: np.ndarray, y: np.ndarray, draws: int = 20000, seed: int = 0) -> float:
    """Two-sided p for a rank correlation, by shuffling the labels.

    Exact enough at this n and free of any distributional assumption, which matters because the lean
    is bounded in [-1, 1] and visibly non-normal.
    """
    rng = np.random.default_rng(seed)
    observed = abs(spearman(x, y))
    hits = sum(abs(spearman(x, rng.permutation(y))) >= observed for _ in range(draws))
    return (hits + 1) / (draws + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("analysis/zsweep.npz"))
    parser.add_argument("--selection", type=Path, default=Path("analysis/sel.json"))
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = np.load(args.data, allow_pickle=True)
    table, arms = payload["table"], [str(a) for a in payload["arms"]]
    layers = [int(x) for x in payload["layers"]]
    selection = json.loads(args.selection.read_text())
    lean = {int(p): float(l) for p, l in zip(selection["pairs"], selection["leans"])}

    base = arms.index("baseline")
    rows = []
    for a, arm in enumerate(arms):
        if not arm.startswith("pair"):
            continue
        pair = int(arm[4:].split("@")[0])
        row = {"pair": pair, "lean": lean[pair]}
        for l, layer in enumerate(layers):
            raw_base = table[base, :, l, :]
            centre, spread = raw_base.mean(0), raw_base.std(0, ddof=1) + 1e-9
            z_base = (raw_base - centre) / spread
            z_arm = (table[a, :, l, :] - centre) / spread
            pooled = np.sqrt((z_base.var(0, ddof=1) + z_arm.var(0, ddof=1)) / 2) + 1e-9
            d = (z_arm.mean(0) - z_base.mean(0)) / pooled
            mask = np.ones(d.shape, dtype=bool); mask[pair] = False
            row[f"on_L{layer}"] = float(d[pair])
            row[f"off_L{layer}"] = int((np.abs(d[mask]) > args.threshold).sum())
            row[f"offmed_L{layer}"] = float(np.median(np.abs(d[mask])))
        rows.append(row)

    lean_v = np.array([r["lean"] for r in rows])
    print(f"{len(rows)} vectors, behavioural lean {lean_v.min():+.3f} .. {lean_v.max():+.3f}")
    print(f"  {int((lean_v < 0).sum())} steer backwards, {int((lean_v > 0.3).sum())} steer strongly\n")

    print(f"{'residual-stream measure':<34}{'Spearman r':>12}{'perm p':>10}   verdict")
    print("-" * 78)
    report = {"n": len(rows), "correlations": {}}
    for layer in layers:
        for key, label in ((f"on_L{layer}", f"on-target displacement, block {layer}"),
                           (f"off_L{layer}", f"off-target count, block {layer}"),
                           (f"offmed_L{layer}", f"off-target median |d|, block {layer}")):
            values = np.array([r[key] for r in rows], dtype=float)
            r = spearman(values, lean_v)
            p = permutation_p(values, lean_v)
            verdict = "PREDICTS" if p < 0.05 else "no relation"
            print(f"{label:<34}{r:>+12.3f}{p:>10.4f}   {verdict}")
            report["correlations"][key] = {"spearman": r, "p": p, "label": label}

    # The sharpest form of the question: can any of these separate the vectors that steer the RIGHT
    # way from the ones that steer the WRONG way? That is the decision a screen would have to make.
    print("\nseparating working (lean > +0.3) from backwards (lean < 0):")
    good = np.array([r["lean"] > 0.3 for r in rows])
    bad = np.array([r["lean"] < 0.0 for r in rows])
    for layer in layers:
        for key in (f"on_L{layer}", f"off_L{layer}"):
            values = np.array([r[key] for r in rows], dtype=float)
            a, b = values[good], values[bad]
            joint = np.concatenate([a, b]); ranks = joint.argsort().argsort().astype(float) + 1
            auc = (ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
            print(f"  {key:<14} working {a.mean():>9.2f}   backwards {b.mean():>9.2f}   AUC {auc:.3f}")
            report["correlations"].setdefault(key, {})["auc_working_vs_backwards"] = float(auc)

    report["rows"] = rows
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
