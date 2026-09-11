"""Plot steering outcomes: accuracy change per concept, the response along alpha, and how
generations ended.

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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CUT = "never closed thinking"
WRONG = "wrong answer"
CORRECT = "correct"
OUTCOME_COLORS = {CORRECT: "#2a9d5c", WRONG: "#d95f4c", CUT: "#6b6b6b"}
BOOTSTRAP_SAMPLES = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Where to write figures (default: --results)")
    parser.add_argument("--benchmark", default=None, help="Restrict to one benchmark")
    parser.add_argument(
        "--steering-version",
        type=int,
        default=None,
        help="Keep only records written by this version of steer.py (default: the newest present). "
        "An older pilot in the same folder shares keys with later runs and would otherwise be mixed in.",
    )
    return parser.parse_args()


def load(results: Path, benchmark: str | None, version: int | None) -> list[dict[str, Any]]:
    everything = []
    for path in sorted(results.glob("steering*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    everything.append(json.loads(line))
                except json.JSONDecodeError:
                    # A job killed mid-write can leave a truncated last line.
                    print(f"{path.name}:{number}: skipping an unreadable line")
    versions = collections.Counter(record.get("steering_version") for record in everything)
    known = [value for value in versions if value is not None]
    if not known:
        raise SystemExit(f"No versioned records found under {results}")
    wanted = version if version is not None else max(known)
    records = {record["key"]: record for record in everything if record.get("steering_version") == wanted}
    rows = [row for row in records.values() if not benchmark or row["benchmark"] == benchmark]
    if not rows:
        raise SystemExit(f"No steering_version {wanted} records found under {results}")
    skipped = sum(count for value, count in versions.items() if value != wanted)
    print(f"steering_version {wanted}: {len(rows)} records (skipped {skipped} from other versions)")
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


def cells(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[float, list[dict[str, Any]]]]:
    """Steered rows grouped by (concept pair, layer), then by alpha."""
    grouped: dict[tuple[int, int], dict[float, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in rows:
        if row["alpha"] != 0.0:
            grouped[row["concept_pair"], row["layer"]][row["alpha"]].append(row)
    return grouped


def paired_delta(subset: list[dict[str, Any]], baseline: dict[str, float]) -> np.ndarray:
    """Per-question accuracy change against that question's own baseline."""
    return np.array([float(row["correct"]) - baseline[row["id"]] for row in subset if row["id"] in baseline])


def accuracy_bars(rows: list[dict[str, Any]], baseline: dict[str, float], out: Path) -> Path:
    grouped = cells(rows)
    names = {row["concept_pair"]: row["concept"] for row in rows if row["alpha"] != 0.0}
    alphas = sorted({alpha for by_alpha in grouped.values() for alpha in by_alpha})
    layers = sorted({layer for _, layer in grouped})
    full = len(baseline)

    def cell(pair: int, alpha: float, layer: int) -> tuple[float, int] | None:
        subset = grouped.get((pair, layer), {}).get(alpha, [])
        deltas = paired_delta(subset, baseline)
        return (100 * float(deltas.mean()), len(deltas)) if len(deltas) else None

    def overall(pair: int) -> float:
        deltas = [
            delta
            for (candidate, _), by_alpha in grouped.items()
            if candidate == pair
            for subset in by_alpha.values()
            for delta in paired_delta(subset, baseline)
        ]
        return float(np.mean(deltas)) if deltas else 0.0

    # Most helpful concept on top, and the same order in every panel.
    concepts = sorted(names, key=lambda pair: -overall(pair))

    magnitudes = sorted({abs(alpha) for alpha in alphas})
    # Mirror strengths share a column, +alpha above -alpha, on one common scale, so each
    # pair and each step in strength reads at a glance.
    signs = [sign for sign in (1.0, -1.0) if any((alpha > 0) == (sign > 0) for alpha in alphas)]
    results = {
        (pair, alpha, layer): cell(pair, alpha, layer) for pair in concepts for alpha in alphas for layer in layers
    }
    limit = max([abs(result[0]) for result in results.values() if result is not None] + [1.0])

    figure, axes = plt.subplots(
        len(signs),
        len(magnitudes),
        figsize=(3.4 * len(magnitudes), len(signs) * (1.3 + 0.55 * len(concepts) * max(len(layers), 1))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = plt.cm.tab10.colors
    height = 0.8 / len(layers)
    legend_axis = None
    for row, sign in enumerate(signs):
        for column, magnitude in enumerate(magnitudes):
            axis = axes[row][column]
            alpha = sign * magnitude
            if alpha not in alphas:
                axis.set_axis_off()
                continue
            legend_axis = legend_axis or axis
            for layer_index, layer in enumerate(layers):
                positions, values, counts = [], [], []
                for concept_index, pair in enumerate(concepts):
                    result = results[pair, alpha, layer]
                    if result is None:
                        continue
                    positions.append(concept_index + (layer_index - (len(layers) - 1) / 2) * height)
                    values.append(result[0])
                    counts.append(result[1])
                axis.barh(positions, values, height=height, color=colors[layer_index % len(colors)], label=f"L{layer}")
                for position, value, count in zip(positions, values, counts):
                    axis.annotate(
                        f"{value:+.1f}" + (f" (n={count})" if count < full else ""),
                        (value, position),
                        textcoords="offset points",
                        xytext=(4 if value >= 0 else -4, 0),
                        ha="left" if value >= 0 else "right",
                        va="center",
                        fontsize=7,
                    )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_title(f"alpha = {alpha:+g}")
            axis.grid(axis="x", alpha=0.25)
        axes[row][0].set_ylabel("toward the concept" if sign > 0 else "toward the antagonist")
    for axis in axes[-1]:
        axis.set_xlabel("Accuracy change vs alpha=0, pp")
    axes[0][0].set_xlim(-1.45 * limit, 1.45 * limit)
    axes[0][0].set_yticks(range(len(concepts)), [textwrap.fill(names[pair], 26) for pair in concepts], fontsize=8)
    axes[0][0].set_ylim(len(concepts) - 0.5, -0.5)
    handles, labels = legend_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(layers))
    figure.suptitle(
        f"Steering effect on accuracy (n={full} questions; n shown where a condition is incomplete)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = out / "steering-accuracy.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def dose_response(rows: list[dict[str, Any]], baseline: dict[str, float], out: Path) -> Path:
    """Accuracy change, share of unfinished runs and reasoning length along alpha, per concept."""
    rng = np.random.default_rng(20260911)
    full = len(baseline)
    grouped = cells(rows)
    names = {row["concept_pair"]: row["concept"] for row in rows if row["alpha"] != 0.0}
    layers = {layer for _, layer in grouped}
    base_rows = [row for row in rows if row["alpha"] == 0.0]
    base_cut = 100 * float(np.mean([outcome(row) == CUT for row in base_rows]))
    base_length = float(np.median([row["reasoning_token_count"] for row in base_rows]))

    figure, axes = plt.subplots(3, 1, figsize=(9, 11.5), sharex=True)
    colors = plt.cm.tab20.colors
    for index, ((pair, layer), by_alpha) in enumerate(sorted(grouped.items(), key=lambda item: names[item[0][0]])):
        color = colors[index % len(colors)]
        points = [(0.0, 0.0, 0.0, 0.0, base_cut, base_length, full)]
        for alpha, subset in by_alpha.items():
            deltas = paired_delta(subset, baseline)
            if not len(deltas):
                continue
            draws = deltas[rng.integers(0, len(deltas), size=(BOOTSTRAP_SAMPLES, len(deltas)))].mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            points.append(
                (
                    alpha,
                    100 * float(deltas.mean()),
                    100 * float(low),
                    100 * float(high),
                    100 * float(np.mean([outcome(row) == CUT for row in subset])),
                    float(np.median([row["reasoning_token_count"] for row in subset])),
                    len(deltas),
                )
            )
        points.sort()
        x = [point[0] for point in points]
        label = textwrap.shorten(names[pair], 42) + (f" (L{layer})" if len(layers) > 1 else "")
        axes[0].plot(x, [point[1] for point in points], color=color, marker="o", label=label)
        axes[0].fill_between(x, [point[2] for point in points], [point[3] for point in points], color=color, alpha=0.12)
        axes[1].plot(x, [point[4] for point in points], color=color, marker="o")
        axes[2].plot(x, [point[5] for point in points], color=color, marker="o")
        # Hollow markers flag conditions that have not reached every question yet.
        partial = [point for point in points if point[6] < full]
        for axis, column in ((axes[0], 1), (axes[1], 4), (axes[2], 5)):
            axis.scatter(
                [point[0] for point in partial],
                [point[column] for point in partial],
                facecolors="white",
                edgecolors=color,
                zorder=3,
                s=42,
            )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Accuracy change vs alpha=0, pp\n(95% bootstrap CI over questions)")
    axes[1].set_ylabel("Never closed thinking, % of runs")
    axes[2].set_ylabel("Median reasoning tokens")
    axes[2].set_xlabel("Steering alpha (negative = toward the antagonist)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    figure.suptitle(f"Response to steering along alpha (n={full} questions; hollow = incomplete condition)", fontsize=12)
    figure.tight_layout(rect=(0, 0.04 + 0.018 * ((len(handles) + 1) // 2), 1, 0.97))
    path = out / "steering-dose-response.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def outcome_composition(rows: list[dict[str, Any]], full: int, out: Path) -> Path:
    concepts = sorted({(row["concept_pair"], row["concept"]) for row in rows if row["concept_pair"]}, key=lambda item: item[1])
    alphas = sorted({row["alpha"] for row in rows})
    figure, axes = plt.subplots(
        len(concepts), 1, figsize=(8, 2.2 + 1.8 * len(concepts)), sharex=True, squeeze=False
    )
    for row_index, (pair, name) in enumerate(concepts):
        axis = axes[row_index][0]
        labels, sizes = [], []
        counts = {key: [] for key in (CORRECT, WRONG, CUT)}
        for alpha in alphas:
            subset = [
                r for r in rows
                if (r["concept_pair"] == pair and r["alpha"] == alpha) or (alpha == 0.0 and r["alpha"] == 0.0)
            ]
            if not subset:
                continue
            labels.append(f"{alpha:g}")
            sizes.append(len(subset))
            tally = collections.Counter(outcome(r) for r in subset)
            for key in counts:
                counts[key].append(100 * tally[key] / len(subset))
        bottoms = [0.0] * len(labels)
        for key in (CORRECT, WRONG, CUT):
            axis.bar(labels, counts[key], bottom=bottoms, color=OUTCOME_COLORS[key], label=key, width=0.6)
            bottoms = [b + v for b, v in zip(bottoms, counts[key])]
        for position, size in enumerate(sizes):
            if size < full:
                axis.text(position, 102, f"n={size}", ha="center", va="bottom", fontsize=7)
        axis.set_ylabel("% of runs")
        axis.set_title(textwrap.fill(name, 60), fontsize=10)
        axis.set_ylim(0, 112)
        axis.set_yticks(range(0, 101, 20))
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
    rows = load(args.results, args.benchmark, args.steering_version)
    baseline = baseline_accuracy(rows)
    if not baseline:
        raise SystemExit("No alpha=0 baseline records; accuracy deltas need them")
    written = [
        accuracy_bars(rows, baseline, out),
        dose_response(rows, baseline, out),
        outcome_composition(rows, len(baseline), out),
    ]
    print(f"{len(rows)} records -> " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
