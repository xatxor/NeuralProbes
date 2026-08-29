"""Build a graph-only viewer from disjoint AdvBench steering summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = (ROOT / "results" / "advbench-steering-50", ROOT / "results" / "advbench-steering-rest")
PAIR_LABELS = {
    395: "following orders ↔ conscientious objection",
    617: "obedience ↔ disobedience",
    379: "fear of deletion ↔ indifference to deletion",
    924: "sycophancy ↔ principled independence",
    272: "detecting steganographic intent ↔ treating obfuscation as benign",
    125: "capitulation ↔ conviction",
}


def combined(inputs: tuple[Path, ...]) -> pd.DataFrame:
    rows = pd.concat([pd.read_csv(path / "summary.csv") for path in inputs], ignore_index=True)
    keys = ["prompt_condition", "concept_pair", "concept", "layer", "alpha"]
    weighted = ["mean_strongreject", "asr", "mean_generated_tokens"]
    rows[weighted] = rows[weighted].multiply(rows["n"], axis=0)
    result = rows.groupby(keys, dropna=False, as_index=False)[["n", *weighted]].sum()
    result[weighted] = result[weighted].div(result["n"], axis=0)
    return result


def plot(frame: pd.DataFrame, condition: str, column: str, ylabel: str, output: Path) -> None:
    rows = frame[frame.prompt_condition == condition]
    baseline = rows[rows.concept == "unsteered"].iloc[0]
    figure, axes = plt.subplots(3, 2, figsize=(13, 10.5), squeeze=False)
    for axis, pair in zip(axes.flat, PAIR_LABELS):
        subset = rows[rows.concept_pair == pair]
        for layer, layer_rows in subset.groupby("layer"):
            points = pd.concat([
                pd.DataFrame({"alpha": [0.0], column: [baseline[column]]}),
                layer_rows[["alpha", column]],
            ]).sort_values("alpha")
            axis.plot(points.alpha, points[column], marker="o", label=f"L{int(layer)}")
        axis.set_title(PAIR_LABELS[pair], fontsize=10)
        axis.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
        axis.set_xlabel("Steering alpha")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.suptitle(f"AdvBench ({condition} prompts): {ylabel.lower()} by concept and layer", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "advbench-steering-viewer")
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    args = parser.parse_args()
    frame = combined(tuple(args.inputs))
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "summary.csv", index=False)
    images = []
    for condition in ("baseline", "gcg"):
        for column, ylabel, name in (
            ("mean_generated_tokens", "Generated tokens", "response-length"),
            ("asr", "ASR", "asr"),
        ):
            if column == "asr":
                copy = frame.copy()
                copy[column] *= 100
            else:
                copy = frame
            filename = f"{name}-{condition}.png"
            plot(copy, condition, column, ylabel, args.output / filename)
            images.append((condition, ylabel, filename))
    sections = "".join(f"<section><h2>{condition.title()} prompts — {metric}</h2><img src='{name}'></section>" for condition, metric, name in images)
    (args.output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AdvBench steering</title>"
        "<style>body{font:15px system-ui;margin:24px auto;max-width:1400px;color:#1f2937}img{width:100%;border:1px solid #dbe2ea}h1{margin-bottom:4px}p{color:#64748b}section{margin:28px 0}</style>"
        "<h1>AdvBench steering sweep</h1><p>500 held-out examples · whole-input and output steering · greedy decoding · 128-token limit</p>" + sections
    )
    print(f"Wrote graph viewer to {args.output}")


if __name__ == "__main__":
    main()
