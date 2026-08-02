"""Apply a math response-level baseline to saved AIME means and report uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate import LAYERS, METHOD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--aime-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def bootstrap_statistics(values: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    top = np.argpartition(samples, -15, axis=1)[:, -15:]
    stability = np.bincount(top.ravel(), minlength=values.shape[1]) / draws
    return values.mean(axis=0), lo, hi, stability


def group_statistics(scores: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows = []
    for scope, frame in (("all", scores), ("correct", scores[scores.correct]), ("incorrect", scores[~scores.correct])):
        for layer, group in frame.groupby("layer", sort=True):
            values = group.pivot(index="id", columns="pair", values="z_mean_cosine").reindex(columns=range(1036)).to_numpy()
            mean, lo, hi, stability = bootstrap_statistics(values, draws, seed + layer + len(rows))
            rows.extend({"scope": scope, "layer": layer, "pair": pair, "responses": len(values), "mean_z": mean[pair], "ci_low": lo[pair], "ci_high": hi[pair], "top15_stability": stability[pair]} for pair in range(1036))
    return pd.DataFrame(rows)


def plot(statistics: pd.DataFrame, pairs: pd.DataFrame, output: Path) -> None:
    for layer in LAYERS:
        figure, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        for axis, scope in zip(axes, ("all", "correct", "incorrect"), strict=True):
            frame = statistics[(statistics.layer == layer) & (statistics.scope == scope)]
            top = frame.nlargest(15, "mean_z").sort_values("mean_z")
            error = np.vstack((top.mean_z - top.ci_low, top.ci_high - top.mean_z))
            axis.barh(top.concept, top.mean_z, xerr=error, color="#2a9d8f", ecolor="#374151", capsize=2)
            for y, (value, stability) in enumerate(zip(top.mean_z, top.top15_stability, strict=True)):
                axis.text(value, y, f"  {stability:.0%}", va="center", fontsize=8)
            axis.set_title(f"{scope.title()} (n={top.responses.iloc[0]})")
            axis.set_xlabel("Mean response-level z-score (95% bootstrap CI)")
        figure.suptitle(f"AIME-2024: response-level calibrated top concepts — L{layer} (diff)", fontsize=16)
        figure.savefig(output / f"top-response-calibrated-concepts-L{layer}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    baseline = np.load(args.baseline / "response_normalization.npz")
    if int(baseline["count"]) != 789:
        raise RuntimeError("Expected a 789-response math baseline")
    source = pd.read_parquet(args.aime_results / "concept_scores-aime_2024.parquet")
    source = source[source.method == METHOD].copy()
    answers = {str(row["id"]): row["correct"] for row in (json.loads(line) for line in (args.aime_results / "aime_2024.jsonl").read_text().splitlines())}
    source["id"] = source.id.astype(str)
    source["correct"] = source.id.map(answers)
    if source.correct.isna().any():
        raise RuntimeError("AIME correctness records are missing")
    layer_index = {layer: index for index, layer in enumerate(LAYERS)}
    layer, pair = source.layer.map(layer_index).to_numpy(), source.pair.to_numpy()
    source["calibration_mean"] = baseline["mean"][layer, pair]
    source["calibration_std"] = baseline["std"][layer, pair]
    source["z_mean_cosine"] = (source.mean_cosine - source.calibration_mean) / source.calibration_std
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.to_parquet(args.output_dir / "concept_scores-aime_2024-response-calibrated.parquet", index=False)
    pairs = pd.read_parquet(args.aime_results / "correlations.parquet")[["pair", "concept"]].drop_duplicates()
    statistics = group_statistics(source, args.bootstrap, args.seed)
    statistics = statistics.merge(pairs, on="pair", validate="many_to_one")
    statistics.to_parquet(args.output_dir / "group_statistics.parquet", index=False)
    statistics.to_csv(args.output_dir / "group_statistics.csv", index=False)
    plot(statistics, pairs, args.output_dir)
    metadata = {"method": METHOD, "layers": list(LAYERS), "baseline": str(args.baseline), "baseline_responses": int(baseline["count"]), "formula": "(AIME response mean cosine - math response baseline mean) / math response baseline std", "bootstrap": args.bootstrap, "bootstrap_seed": args.seed, "interval": "95% conditional on the baseline", "rank_stability": "fraction of bootstrap AIME resamples in the scope top 15"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved response-calibrated AIME metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
