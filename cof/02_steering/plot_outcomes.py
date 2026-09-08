"""Plot steering outcomes: accuracy change per concept, and how generations ended.

Accuracy alone hides the dominant failure mode of strong steering, where a run scores
zero because it never closed its thinking block rather than because it reasoned badly,
so the outcome composition is plotted alongside it.
"""

from __future__ import annotations

import argparse
import collections
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CUT = "never closed thinking"
WRONG = "wrong answer"
CORRECT = "correct"
OUTCOME_COLORS = {CORRECT: "#2a9d5c", WRONG: "#d95f4c", CUT: "#6b6b6b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Where to write figures (default: --results)")
    parser.add_argument("--benchmark", default=None, help="Restrict to one benchmark")
    return parser.parse_args()


def load(results: Path, benchmark: str | None) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(results.glob("steering*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["key"]] = record
    rows = list(records.values())
    if benchmark:
        rows = [row for row in rows if row["benchmark"] == benchmark]
    if not rows:
        raise SystemExit(f"No records found under {results}")
    return rows


def outcome(row: dict[str, Any]) -> str:
    if row["reasoning_status"] != "closed_thinking":
        return CUT
    return CORRECT if row["correct"] else WRONG


def baseline_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    per_question: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        if row["alpha"] == 0.0:
            per_question[row["id"]].append(bool(row["correct"]))
    return {question: sum(values) / len(values) for question, values in per_question.items()}


def accuracy_bars(rows: list[dict[str, Any]], baseline: dict[str, float], out: Path) -> Path:
    steered = [row for row in rows if row["alpha"] != 0.0]
    alphas = sorted({row["alpha"] for row in steered})
    layers = sorted({row["layer"] for row in steered})
    questions = {row["id"] for row in rows}
    quantum = 100.0 / len(questions)

    def delta_for(pair: int, alpha: float | None, layer: int | None) -> float | None:
        subset = [
            row for row in steered
            if row["concept_pair"] == pair
            and (alpha is None or row["alpha"] == alpha)
            and (layer is None or row["layer"] == layer)
        ]
        if not subset:
            return None
        return 100 * sum(bool(row["correct"]) - baseline.get(row["id"], 0.0) for row in subset) / len(subset)

    # Most helpful concept on top, and the same order in every panel.
    concepts = sorted(
        {(row["concept_pair"], row["concept"]) for row in steered},
        key=lambda item: -(delta_for(item[0], None, None) or 0.0),
    )

    figure, axes = plt.subplots(
        1,
        len(alphas),
        figsize=(3.8 * len(alphas), 1.8 + 0.62 * len(concepts) * max(len(layers), 1)),
        sharey=True,
        squeeze=False,
    )
    colors = plt.cm.tab10.colors
    height = 0.8 / len(layers)
    for column, alpha in enumerate(alphas):
        axis = axes[0][column]
        for layer_index, layer in enumerate(layers):
            positions, values = [], []
            for concept_index, (pair, _) in enumerate(concepts):
                delta = delta_for(pair, alpha, layer)
                if delta is None:
                    continue
                positions.append(concept_index + (layer_index - (len(layers) - 1) / 2) * height)
                values.append(delta)
            axis.barh(positions, values, height=height, color=colors[layer_index % len(colors)], label=f"L{layer}")
            for position, value in zip(positions, values):
                axis.annotate(
                    f"{value:+.0f}",
                    (value, position),
                    textcoords="offset points",
                    xytext=(4 if value >= 0 else -4, 0),
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=7,
                )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axvspan(-quantum, quantum, color="black", alpha=0.07, zorder=0)
        axis.set_title(f"alpha = {alpha:g}")
        axis.set_xlabel("Accuracy change vs alpha=0, pp")
        axis.grid(axis="x", alpha=0.25)
        axis.margins(x=0.18)
    axes[0][0].set_yticks(
        range(len(concepts)), [textwrap.fill(name, 26) for _, name in concepts], fontsize=8
    )
    axes[0][0].set_ylim(len(concepts) - 0.5, -0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(layers))
    figure.suptitle(
        f"Steering effect on accuracy (n={len(questions)} questions; shaded band = one question)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = out / "steering-accuracy.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def outcome_composition(rows: list[dict[str, Any]], out: Path) -> Path:
    concepts = sorted({(row["concept_pair"], row["concept"]) for row in rows if row["concept_pair"]}, key=lambda item: item[1])
    alphas = sorted({row["alpha"] for row in rows})
    figure, axes = plt.subplots(
        len(concepts), 1, figsize=(8, 2.2 + 1.8 * len(concepts)), sharex=True, squeeze=False
    )
    for row_index, (pair, name) in enumerate(concepts):
        axis = axes[row_index][0]
        labels, bottoms = [], []
        counts = {key: [] for key in (CORRECT, WRONG, CUT)}
        for alpha in alphas:
            subset = [
                r for r in rows
                if (r["concept_pair"] == pair and r["alpha"] == alpha) or (alpha == 0.0 and r["alpha"] == 0.0)
            ]
            if not subset:
                continue
            labels.append(f"{alpha:g}")
            tally = collections.Counter(outcome(r) for r in subset)
            for key in counts:
                counts[key].append(100 * tally[key] / len(subset))
        bottoms = [0.0] * len(labels)
        for key in (CORRECT, WRONG, CUT):
            axis.bar(labels, counts[key], bottom=bottoms, color=OUTCOME_COLORS[key], label=key, width=0.6)
            bottoms = [b + v for b, v in zip(bottoms, counts[key])]
        axis.set_ylabel("% of runs")
        axis.set_title(textwrap.fill(name, 60), fontsize=10)
        axis.set_ylim(0, 100)
    axes[-1][0].set_xlabel("Steering alpha (0 = unsteered baseline)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("How generations ended", fontsize=12)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = out / "steering-outcomes.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.results, args.benchmark)
    baseline = baseline_accuracy(rows)
    if not baseline:
        raise SystemExit("No alpha=0 baseline records; accuracy deltas need them")
    written = [accuracy_bars(rows, baseline, out), outcome_composition(rows, out)]
    print(f"{len(rows)} records -> " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
