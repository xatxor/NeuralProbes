"""Build summary plots and a static index for the projection-probe viewer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from pipeline import LAYERS, PAIRS, VECTOR_REPO, VECTOR_REVISION

ROOT = Path(__file__).resolve().parent


def bootstrap_statistics(
    values: np.ndarray, draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975], axis=0)
    return values.mean(axis=0), low, high


def group_statistics(scores: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows = []
    scopes = (
        ("all", scores),
        ("correct", scores[scores.correct]),
        ("incorrect", scores[~scores.correct]),
    )
    for scope_index, (scope, frame) in enumerate(scopes):
        for layer in LAYERS:
            group = frame[frame.layer == layer]
            values = (
                group.pivot(index="id", columns="pair", values="mean_projection")
                .reindex(columns=range(PAIRS))
                .to_numpy(dtype=np.float32)
            )
            mean, low, high = bootstrap_statistics(
                values, draws, seed + scope_index * 100 + layer
            )
            rows.extend(
                {
                    "scope": scope,
                    "layer": layer,
                    "pair": pair,
                    "responses": len(values),
                    "mean_projection": float(mean[pair]),
                    "ci_low": float(low[pair]),
                    "ci_high": float(high[pair]),
                }
                for pair in range(PAIRS)
            )
    return pd.DataFrame(rows)


def plots(statistics: pd.DataFrame, output: Path) -> None:
    for layer in LAYERS:
        figure, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)
        for axis, scope in zip(axes, ("all", "correct", "incorrect"), strict=True):
            frame = statistics[
                (statistics.layer == layer) & (statistics.scope == scope)
            ]
            top = frame.nlargest(15, "mean_projection").sort_values("mean_projection")
            errors = np.vstack(
                (
                    top.mean_projection - top.ci_low,
                    top.ci_high - top.mean_projection,
                )
            )
            axis.barh(
                top.concept,
                top.mean_projection,
                xerr=errors,
                color="#b91c1c",
                alpha=0.82,
                ecolor="#374151",
                capsize=2,
            )
            axis.axvline(0, color="#6b7280", linewidth=0.8)
            axis.set_title(f"{scope.title()} (n={top.responses.iloc[0]})")
            axis.set_xlabel("Mean raw projection on thinking tokens (95% bootstrap CI)")
        figure.suptitle(
            f"AIME-2024 article-style diff probes — L{layer}", fontsize=16
        )
        figure.savefig(output / f"top-projections-L{layer}.png", dpi=180)
        plt.close(figure)


def build_viewer(scores: pd.DataFrame, pairs: pd.DataFrame, output: Path) -> None:
    viewer = output / "viewer"
    viewer.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "viewer.html", viewer / "index.html")
    responses = (
        scores[["id", "correct", "reasoning_tokens"]]
        .drop_duplicates()
        .sort_values("id", key=lambda column: column.astype(int))
        .to_dict("records")
    )
    means = {}
    rankings = {}
    for layer in LAYERS:
        layer_scores = scores[scores.layer == layer]
        rankings[str(layer)] = (
            layer_scores.groupby("pair").mean_projection.mean().sort_values(ascending=False).index.tolist()
        )
        for response_id, group in layer_scores.groupby("id"):
            means[f"{response_id}:L{layer}"] = (
                group.set_index("pair").mean_projection.reindex(range(PAIRS)).astype(float).tolist()
            )
    index = {
        "layers": list(LAYERS),
        "concepts": pairs[["pair", "concept", "antagonist", "class_name"]].to_dict("records"),
        "responses": responses,
        "means": means,
        "rankings": rankings,
        "description": (
            "Raw residual-stream projection onto unit diff vectors after removing "
            "OpenThoughts math background PCs. Rankings average thinking tokens; "
            "token colors use the full saved output's 99th-percentile magnitude."
        ),
    }
    (viewer / "index.json").write_text(json.dumps(index, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    scores = pd.read_parquet(args.results / "concept_scores-aime_2024.parquet")
    pairs = pd.read_parquet(
        hf_hub_download(VECTOR_REPO, "pairs.parquet", revision=VECTOR_REVISION)
    )[["pair", "concept", "antagonist", "class_name"]]
    statistics = group_statistics(scores, args.bootstrap, args.seed).merge(
        pairs, on="pair", validate="many_to_one"
    )
    statistics.to_parquet(args.results / "group_statistics.parquet", index=False)
    statistics.to_csv(args.results / "group_statistics.csv", index=False)
    plots(statistics, args.results)
    build_viewer(scores, pairs, args.results)
    print(f"Built plots and viewer in {args.results}", flush=True)


if __name__ == "__main__":
    main()
