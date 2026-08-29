"""Summarize steering effects against the shared alpha-zero baseline."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steer import RESULTS, STEERING_VERSION, VECTOR_REVISION, iter_records  # noqa: E402

BOOTSTRAP_SAMPLES = 2_000


def mean_ci(group: pd.DataFrame, column: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Question-level 95% bootstrap interval; repeated baseline runs count once."""
    values = group.groupby("id")[column].mean().to_numpy(dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def metric_row(group: pd.DataFrame) -> dict:
    return {
        "n": len(group),
        "mean_reasoning_tokens": group.reasoning_token_count.mean(),
        "accuracy_pct": group.correct.mean() * 100,
        "mean_generation_seconds": group.generation_seconds.mean(),
        "context_limit_pct": group.hit_context_limit.mean() * 100,
    }


def baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.alpha == 0.0]
    if "baseline_repeat" in baseline and baseline.baseline_repeat.notna().any():
        return baseline[baseline.baseline_repeat.notna()]
    return baseline


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_frame = baseline_rows(frame)
    baseline = baseline_frame.groupby(["benchmark", "id"])[["reasoning_token_count", "correct"]].mean()
    summary_rows = []
    for benchmark, group in frame.groupby("benchmark"):
        summary_rows.append(
            {"benchmark": benchmark, "concept_pair": None, "concept": "baseline", "layer": None, "alpha": 0.0, **metric_row(baseline_rows(group))}
        )
    effects = []
    nonzero = frame[frame.alpha != 0.0]
    for keys, group in nonzero.groupby(["benchmark", "concept_pair", "concept", "layer", "alpha"], dropna=False):
        benchmark, pair, concept, layer, alpha = keys
        summary_rows.append(
            {"benchmark": benchmark, "concept_pair": pair, "concept": concept, "layer": layer, "alpha": alpha, **metric_row(group)}
        )
        paired = group.set_index(["benchmark", "id"]).join(
            baseline[["reasoning_token_count", "correct"]],
            how="inner",
            rsuffix="_baseline",
        )
        current_tokens = paired.reasoning_token_count.mean()
        baseline_tokens = paired.reasoning_token_count_baseline.mean()
        effects.append(
            {
                "benchmark": benchmark,
                "concept_pair": pair,
                "concept": concept,
                "layer": layer,
                "alpha": alpha,
                "n_paired": len(paired),
                "mean_reasoning_tokens": current_tokens,
                "baseline_mean_reasoning_tokens": baseline_tokens,
                "delta_reasoning_tokens": current_tokens - baseline_tokens,
                "delta_reasoning_pct": (current_tokens / baseline_tokens - 1) * 100 if baseline_tokens else None,
                "accuracy_pct": paired.correct.mean() * 100,
                "baseline_accuracy_pct": paired.correct_baseline.mean() * 100,
                "delta_accuracy_pp": (paired.correct.mean() - paired.correct_baseline.mean()) * 100,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(effects)


def plot_results(frame: pd.DataFrame) -> list[Path]:
    paths = []
    rng = np.random.default_rng(20260729)
    for benchmark, rows in frame.groupby("benchmark"):
        baseline = baseline_rows(rows)
        steered = rows[rows.alpha != 0.0]
        concepts = list(steered[["concept_pair", "concept"]].drop_duplicates().itertuples(index=False, name=None))
        if not concepts or baseline.empty:
            continue
        alphas = sorted(set(steered.alpha) | {0.0})
        alpha_positions = {alpha: index for index, alpha in enumerate(alphas)}
        columns = 2
        figure, axes = plt.subplots((len(concepts) + 1) // columns, columns, figsize=(13, 3.4 * ((len(concepts) + 1) // columns)), squeeze=False)
        baseline_tokens, baseline_low, baseline_high = mean_ci(baseline, "reasoning_token_count", rng)
        for axis, (pair, concept) in zip(axes.flat, concepts):
            subset = steered[steered.concept_pair == pair]
            for layer, layer_rows in subset.groupby("layer"):
                points = [(0.0, baseline_tokens, baseline_low, baseline_high, baseline.hit_context_limit.any())]
                points.extend(
                    (alpha, *mean_ci(group, "reasoning_token_count", rng), group.hit_context_limit.any())
                    for alpha, group in layer_rows.groupby("alpha")
                )
                points.sort()
                positions = [alpha_positions[point[0]] for point in points]
                values = [point[1] for point in points]
                line, = axis.plot(positions, values, marker="o", label=f"L{int(layer)}")
                axis.fill_between(positions, [point[2] for point in points], [point[3] for point in points], color=line.get_color(), alpha=0.18)
                limited = [point for point in points if point[4]]
                if limited:
                    axis.scatter([alpha_positions[point[0]] for point in limited], [point[1] for point in limited], marker="x", s=70, color="black")
            axis.set_title(textwrap.fill(concept, 34))
            axis.set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
            axis.set_xlabel("Steering alpha")
            axis.set_ylabel("Reasoning tokens")
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(concepts):]:
            axis.remove()
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.suptitle(f"{benchmark}: reasoning length by concept and layer (× = context limit)", y=0.995)
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.976), ncol=len(labels), fontsize="small")
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        path = RESULTS / f"reasoning-length-{benchmark}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

        layers = sorted(steered.layer.dropna().unique())
        figure, axes = plt.subplots((len(concepts) + 1) // columns, columns, figsize=(13, 3.2 * ((len(concepts) + 1) // columns)), squeeze=False)
        baseline_accuracy, baseline_accuracy_low, baseline_accuracy_high = mean_ci(baseline, "correct", rng)
        baseline_accuracy *= 100
        baseline_accuracy_low *= 100
        baseline_accuracy_high *= 100
        for axis, (pair, concept) in zip(axes.flat, concepts):
            values, intervals = [], []
            subset = steered[steered.concept_pair == pair]
            for layer in layers:
                row, row_intervals = [], []
                for alpha in alphas:
                    condition = subset[(subset.layer == layer) & (subset.alpha == alpha)]
                    if alpha == 0.0:
                        value, low, high = baseline_accuracy, baseline_accuracy_low, baseline_accuracy_high
                    else:
                        value, low, high = mean_ci(condition, "correct", rng)
                        value, low, high = value * 100, low * 100, high * 100
                    row.append(value)
                    row_intervals.append((low, high))
                values.append(row)
                intervals.append(row_intervals)
            image = axis.imshow(values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
            for y, row in enumerate(values):
                for x, value in enumerate(row):
                    low, high = intervals[y][x]
                    axis.text(x, y, f"{value:.0f}%\n[{low:.0f}, {high:.0f}]", ha="center", va="center", fontsize=6)
            axis.set_title(textwrap.fill(concept, 34))
            axis.set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
            axis.set_yticks(range(len(layers)), [f"L{int(layer)}" for layer in layers])
            axis.set_xlabel("Steering alpha")
        for axis in axes.flat[len(concepts):]:
            axis.remove()
        colorbar_axis = figure.add_axes([0.92, 0.15, 0.015, 0.7])
        figure.colorbar(image, cax=colorbar_axis, label="Accuracy (%)")
        figure.suptitle(f"{benchmark}: correctness by concept, layer, and alpha (95% bootstrap CI)", y=0.995)
        figure.subplots_adjust(top=0.95, bottom=0.07, left=0.08, right=0.88, hspace=0.55, wspace=0.25)
        path = RESULTS / f"accuracy-{benchmark}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> None:
    files = sorted(RESULTS.glob("steering*.jsonl"))
    columns = (
        "key", "benchmark", "id", "concept_pair", "concept", "layer", "alpha", "baseline_repeat",
        "correct", "reasoning_token_count", "generation_seconds", "hit_context_limit",
    )
    records = {}
    for path in files:
        for row in iter_records(path):
            if row.get("steering_version") == STEERING_VERSION and row.get("vector_revision") == VECTOR_REVISION:
                records[row["key"]] = {column: row.get(column) for column in columns}
    if not records:
        raise SystemExit(f"No compatible records found under {RESULTS}")
    frame = pd.DataFrame(records.values())
    summary, effects = summarize(frame)
    summary.sort_values(["benchmark", "concept_pair", "layer", "alpha"], na_position="first").to_csv(RESULTS / "summary.csv", index=False)
    if effects.empty:
        effects.to_csv(RESULTS / "effects.csv", index=False)
    else:
        effects.sort_values(["benchmark", "concept_pair", "layer", "alpha"]).to_csv(RESULTS / "effects.csv", index=False)
    plots = plot_results(frame)
    print(f"Wrote summary tables and {len(plots)} plots under {RESULTS} from {len(records)} generations")


if __name__ == "__main__":
    main()
