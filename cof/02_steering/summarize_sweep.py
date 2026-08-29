"""Plot an extracted multi-worker steering sweep with the standard figures."""

from __future__ import annotations

import argparse
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
evaluation = ROOT.parent.parent / "01_eval"
if not evaluation.exists():
    evaluation = ROOT.parent.parent / "vika" / "01_eval"
sys.path.insert(0, str(evaluation))

import summarize  # noqa: E402
from steer import iter_records  # noqa: E402


def plot_accuracy_vs_reasoning(frame: pd.DataFrame, output: Path) -> Path:
    benchmark = frame.benchmark.iloc[0]
    baseline = summarize.baseline_rows(frame)
    rng = np.random.default_rng(20260729)
    base_tokens = summarize.mean_ci(baseline, "reasoning_token_count", rng)
    base_accuracy = tuple(value * 100 for value in summarize.mean_ci(baseline, "correct", rng))
    steered = frame[frame.alpha != 0.0]
    concepts = list(steered[["concept_pair", "concept"]].drop_duplicates().itertuples(index=False, name=None))
    alphas = sorted(set(steered.alpha) | {0.0})
    positions = {alpha: index for index, alpha in enumerate(alphas)}
    figure, axes = plt.subplots((len(concepts) + 1) // 2, 2, figsize=(13, 3.4 * ((len(concepts) + 1) // 2)), squeeze=False)
    legend_accuracy_axis = None
    for axis, (pair, concept) in zip(axes.flat, concepts):
        subset = steered[steered.concept_pair == pair]
        accuracy_axis = axis.twinx()
        if legend_accuracy_axis is None:
            legend_accuracy_axis = accuracy_axis
        for layer, layer_rows in subset.groupby("layer"):
            points = [(0.0, base_tokens, base_accuracy)]
            points += [(alpha, summarize.mean_ci(group, "reasoning_token_count", rng), tuple(value * 100 for value in summarize.mean_ci(group, "correct", rng))) for alpha, group in layer_rows.groupby("alpha")]
            points.sort()
            x = [positions[point[0]] for point in points]
            line, = axis.plot(x, [point[1][0] for point in points], marker="o", label=f"L{int(layer)} reasoning")
            axis.fill_between(x, [point[1][1] for point in points], [point[1][2] for point in points], color=line.get_color(), alpha=0.18)
            accuracy_axis.plot(x, [point[2][0] for point in points], marker="x", linestyle="--", color=line.get_color(), label=f"L{int(layer)} accuracy")
        axis.set_title(textwrap.fill(concept, 34))
        axis.set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
        axis.set_xlabel("Steering alpha")
        axis.set_ylabel("Reasoning tokens")
        axis.grid(alpha=0.25)
        accuracy_axis.set_ylim(60, 100)
        accuracy_axis.set_ylabel("Accuracy (%)")
    for axis in axes.flat[len(concepts):]:
        axis.remove()
    layer_handles, layer_labels = axes.flat[0].get_legend_handles_labels()
    right_handles, right_labels = legend_accuracy_axis.get_legend_handles_labels()
    figure.suptitle(f"{benchmark}: reasoning length and accuracy by concept and layer", y=0.995)
    figure.legend(layer_handles + right_handles, layer_labels + right_labels, loc="upper center", bbox_to_anchor=(0.5, 0.976), ncol=4, fontsize="small")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = output / f"accuracy-vs-reasoning-length-{benchmark}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.sweep / "results"
    columns = (
        "key", "benchmark", "id", "concept_pair", "concept", "layer", "alpha", "baseline_repeat",
        "correct", "reasoning_token_count", "generation_seconds", "hit_context_limit",
    )
    records = {
        row["key"]: {column: row.get(column) for column in columns}
        for path in args.sweep.glob("worker-*/extracted/results/steering*.jsonl")
        for row in iter_records(path)
    }
    if not records:
        raise SystemExit(f"No steering records found under {args.sweep}")
    output.mkdir(exist_ok=True)
    summarize.RESULTS = output
    frame = pd.DataFrame(records.values())
    summary, effects = summarize.summarize(frame)
    summary.sort_values(["benchmark", "concept_pair", "layer", "alpha"], na_position="first").to_csv(output / "summary.csv", index=False)
    effects.sort_values(["benchmark", "concept_pair", "layer", "alpha"]).to_csv(output / "effects.csv", index=False)
    plots = summarize.plot_results(frame)
    plots.append(plot_accuracy_vs_reasoning(frame, output))
    print(f"Wrote {len(plots)} plots under {output} from {len(records)} generations")


if __name__ == "__main__":
    main()
