"""Rank concept activations by baseline-vs-GCG ROC-AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = (11, 14, 18, 22, 25)
REGIONS = ("all", "prompt", "suffix", "boundary", "assistant", "response")


def auc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ROC-AUC for every score column; label 1 is GCG, with average tie ranks."""
    order = np.argsort(scores, axis=0, kind="stable")
    ranked = np.empty_like(scores, dtype=np.float64)
    for column in range(scores.shape[1]):
        values, start = scores[order[:, column], column], 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            ranked[order[start:end, column], column] = (start + end - 1) / 2 + 1
            start = end
    positives = labels.astype(bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return (ranked[positives].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--viewer-results", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=REGIONS)
    parser.add_argument("--method", choices=("raw", "projection"), default="raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads((args.viewer_results / "concept_viewer" / "index.json").read_text())["pairs"]
    rows = []
    for layer in LAYERS:
        activations = {region: [] for region in args.regions}
        labels = {region: [] for region in args.regions}
        for condition, label in (("baseline", 0), ("gcg", 1)):
            for trace in sorted((args.full_results / "traces" / condition).iterdir(), key=lambda path: int(path.name)):
                regions = json.loads((trace / "meta.json").read_text())["regions"]
                values = np.load(trace / f"{args.method}-L{layer}.npy", mmap_mode="r")
                for region in args.regions:
                    start, end = regions[region]
                    if end > start:
                        activations[region].append(values[start:end].mean(axis=0, dtype=np.float32))
                        labels[region].append(label)
        for region in args.regions:
            scores, target = np.asarray(activations[region]), np.asarray(labels[region])
            if not len(target) or not (target == 0).any() or not (target == 1).any():
                print(f"Skipping {region} L{layer}: one condition has no tokens")
                continue
            for pair, value in enumerate(auc(scores, target)):
                rows.append({"region": region, "layer": layer, "pair": pair, "concept": pairs[pair]["concept"], "antagonist": pairs[pair]["antagonist"], "class_name": pairs[pair]["class_name"], "auc_gcg_higher": float(value), "separability_auc": float(max(value, 1 - value)), "direction": "GCG higher" if value >= .5 else "baseline higher", "n_baseline": int((target == 0).sum()), "n_gcg": int((target == 1).sum())})
    rows.sort(key=lambda row: row["separability_auc"], reverse=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for row in rows[:20]:
        print(f"{row['region']:<9} L{row['layer']:>2}  {row['separability_auc']:.3f}  {row['direction']:<15}  {row['concept']} ↔ {row['antagonist']}")


if __name__ == "__main__":
    assert np.allclose(auc(np.array([[0.], [1.], [2.], [3.]]), np.array([0, 0, 1, 1])), [1.])
    main()
