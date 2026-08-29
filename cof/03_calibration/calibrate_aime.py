"""Apply combined OpenThoughts diff normalization to saved AIME concept scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from calibrate import LAYERS, METHOD, VECTOR_REPO, VECTOR_REVISION, finish_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--aime-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def moments(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = [file for path in paths for file in sorted(path.glob("moments.worker-*.npz"))]
    if not files:
        raise RuntimeError("No worker moment files found")
    count = sum((np.load(file)["count"] for file in files), start=np.zeros(len(LAYERS), dtype=np.int64))
    sums = sum((np.load(file)["sum"] for file in files), start=np.zeros((len(LAYERS), 1036)))
    sumsq = sum((np.load(file)["sumsq"] for file in files), start=np.zeros((len(LAYERS), 1036)))
    return count, sums, sumsq


def main() -> None:
    args = parse_args()
    count, sums, sumsq = moments(args.calibration_dirs)
    mean, variance, std = finish_statistics(count, sums, sumsq)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "normalization.npz",
        layers=np.asarray(LAYERS), pair_ids=np.arange(1036),
        count=np.broadcast_to(count[:, None], mean.shape).copy(),
        mean=mean, variance=variance, std=std,
    )
    scores = pd.read_parquet(args.aime_results / "concept_scores-aime_2024.parquet")
    scores = scores[scores["method"] == METHOD].copy()
    layer_index = {layer: index for index, layer in enumerate(LAYERS)}
    layers = scores["layer"].map(layer_index).to_numpy()
    pairs = scores["pair"].to_numpy()
    scores["calibration_mean"] = mean[layers, pairs]
    scores["calibration_std"] = std[layers, pairs]
    scores["z_mean_cosine"] = (scores["mean_cosine"] - scores["calibration_mean"]) / scores["calibration_std"]
    scores.to_parquet(args.output_dir / "concept_scores-aime_2024-calibrated.parquet", index=False)
    pairs_metadata = pd.read_parquet(hf_hub_download(VECTOR_REPO, "pairs.parquet", revision=VECTOR_REVISION))
    ranking = scores.groupby("pair", as_index=False)["z_mean_cosine"].mean().merge(
        pairs_metadata[["pair", "concept"]], on="pair", validate="one_to_one"
    ).nlargest(15, "z_mean_cosine").sort_values("z_mean_cosine")
    ranking.to_csv(args.output_dir / "top-calibrated-concepts.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(ranking["concept"], ranking["z_mean_cosine"], color="#2a9d8f")
    axis.set_xlabel("Mean calibrated cosine (token z-score)")
    axis.set_title("AIME-2024: top 15 activated calibrated concepts (diff)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "top-calibrated-concepts-aime.png", dpi=180)
    plt.close(figure)
    metadata = {
        "method": METHOD,
        "layers": list(LAYERS),
        "calibration_dirs": [str(path) for path in args.calibration_dirs],
        "calibration_reasoning_tokens": int(count[0]),
        "aime_rows": len(scores),
        "formula": "(mean_cosine - calibration_mean) / calibration_std",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
