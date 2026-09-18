"""Plot steering outcomes: accuracy change per concept, the response along alpha, and how
generations ended.

Accuracy alone hides the dominant failure mode of strong steering, where a run scores
zero because it never closed its thinking block rather than because it reasoned badly,
so the outcome composition is plotted alongside it. Reasoning length is shown apart for
right and wrong answers: unfinished runs sit at the token budget and would otherwise drag
a single median toward it.

Labels are in Russian, the language the results are reported in.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CUT = "не закрыл рассуждение, ответа нет"
WRONG = "неверный ответ"
CORRECT = "верный ответ"
OUTCOME_COLORS = {CORRECT: "#2a9d5c", WRONG: "#d95f4c", CUT: "#6b6b6b"}
BOOTSTRAP_SAMPLES = 2_000
# A median over a handful of answers is noise, so such points are left off the length panels.
MIN_ANSWERS = 10
# Fixed per concept, so a concept keeps its colour when others finish and join the plot.
CONCEPT_COLORS = {657: "#2a78d6", 532: "#eb6834", 367: "#1baf7a", 459: "#eda100", 703: "#4a3aa7"}
FALLBACK_COLORS = ("#e87ba4", "#008300", "#e34948")
BENCHMARK_NAMES = {"gpqa_diamond": "GPQA Diamond", "aime_2024": "AIME 2024", "math_500": "MATH-500"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Where to write figures (default: --results)")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Restrict to one or more comma-separated benchmarks; omit or use 'all' to pool all present",
    )
    parser.add_argument(
        "--steering-version",
        type=int,
        default=None,
        help="Keep only records written by this version of steer.py (default: the newest present). "
        "An older pilot in the same folder shares keys with later runs and would otherwise be mixed in.",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=None,
        help="Vector directory. Given it, the dose-response x axis is converted from alpha into "
        "fractions of the residual-stream norm, the units the emotion-vector paper reports.",
    )
    parser.add_argument(
        "--residual-norm",
        type=float,
        default=92.0,
        help="Average residual-stream norm at the steered layer; only used with --vector-dir.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Also plot concepts that have not reached every question at every strength.",
    )
    parser.add_argument(
        "--reasoning-scope",
        choices=("all", "closed"),
        default="all",
        help="Use all generations (unfinished count as failures) or condition on a closed thinking block.",
    )
    return parser.parse_args()


def benchmark_selection(value: str | None) -> tuple[str, ...] | None:
    """Turn the CLI spelling into a benchmark filter; None means pool everything present."""
    if value is None or value.strip().lower() == "all":
        return None
    selected = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not selected:
        raise ValueError("--benchmark must name at least one benchmark")
    return selected


def benchmark_label(rows: list[dict[str, Any]]) -> str:
    """Human-readable label for a single benchmark or a pooled set."""
    names = sorted({row["benchmark"] for row in rows})
    return " + ".join(BENCHMARK_NAMES.get(name, name) for name in names)


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
    records = {
        record["key"]: dict(record)
        for record in everything
        if record.get("steering_version") == wanted
    }
    overrides_path = results / "math-correctness.json"
    applied = 0
    if overrides_path.exists():
        payload = json.loads(overrides_path.read_text(encoding="utf-8"))
        if payload.get("steering_version") != wanted:
            raise SystemExit(
                f"{overrides_path} is for steering_version {payload.get('steering_version')}, expected {wanted}"
            )
        for key, value in payload.get("scores", {}).items():
            if key not in records or records[key].get("correct") is not None:
                continue
            if isinstance(value, dict):
                fingerprint = hashlib.sha256(records[key]["output"].encode("utf-8")).hexdigest()
                if value.get("output_sha256") != fingerprint:
                    continue
                value = value["correct"]
            records[key]["correct"] = bool(value)
            applied += 1
    selected = benchmark_selection(benchmark)
    rows = [
        row for row in records.values()
        if selected is None or row["benchmark"] in selected
    ]
    if not rows:
        raise SystemExit(f"No steering_version {wanted} records found under {results}")
    missing_scores = sum(row.get("correct") is None for row in rows)
    if missing_scores:
        raise SystemExit(
            f"{missing_scores} selected records have correct=null. Run rescore_math.py --results {results} first."
        )
    skipped = sum(count for value, count in versions.items() if value != wanted)
    corrected = f", applied {applied} MATH score overrides" if applied else ""
    print(f"steering_version {wanted}: {len(rows)} records (skipped {skipped} from other versions{corrected})")
    return rows


def pair_table(vector_dir: Path):
    """Read either historical ``pair_id`` or published ``pair`` metadata."""
    import pandas as pd

    table = pd.read_parquet(vector_dir / "pairs.parquet")
    identifier = "pair_id" if "pair_id" in table.columns else "pair"
    if identifier not in table.columns:
        raise ValueError(f"{vector_dir / 'pairs.parquet'} has neither pair_id nor pair")
    return table.set_index(identifier, drop=False)


def outcome(row: dict[str, Any]) -> str:
    if row["reasoning_status"] != "closed_thinking":
        return CUT
    return CORRECT if row["correct"] else WRONG


def concept_color(pair: int) -> str:
    return CONCEPT_COLORS.get(pair, FALLBACK_COLORS[pair % len(FALLBACK_COLORS)])


def question_key(row: dict[str, Any]) -> tuple[str, str]:
    """Questions are numbered from zero within each benchmark, so the name belongs in the key."""
    return row["benchmark"], row["id"]


def baseline_accuracy(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    per_question: dict[tuple[str, str], list[bool]] = collections.defaultdict(list)
    for row in rows:
        if row["alpha"] == 0.0:
            per_question[question_key(row)].append(bool(row["correct"]))
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


def paired_delta(subset: list[dict[str, Any]], baseline: dict[tuple[str, str], float]) -> np.ndarray:
    """Per-question accuracy change against that question's own baseline."""
    return np.array(
        [float(row["correct"]) - baseline[question_key(row)] for row in subset if question_key(row) in baseline]
    )


def only_complete(rows: list[dict[str, Any]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    """Drop concepts that have not reached every question at every strength.

    A half-finished condition still draws a bar and a point, and at a few dozen questions
    that bar is noise wide enough to read as an effect.
    """
    full = len(baseline)
    alphas = sorted({row["alpha"] for row in rows if row["alpha"] != 0.0})
    keep = set()
    for key, by_alpha in cells(rows).items():
        counts = [len(by_alpha.get(alpha, [])) for alpha in alphas]
        name = next(row["concept"] for subset in by_alpha.values() for row in subset)
        if all(count == full for count in counts):
            keep.add(key)
        else:
            print(f"skipping {name}: {sum(counts)}/{full * len(alphas)} generations")
    if not keep:
        raise SystemExit("No concept is finished yet; pass --include-partial to plot anyway")
    return [row for row in rows if row["alpha"] == 0.0 or (row["concept_pair"], row["layer"]) in keep]


def strength_scale(vector_dir: Path | None, residual_norm: float, rows: list[dict[str, Any]]) -> dict[tuple[int, int], float] | None:
    """Per concept, the factor turning alpha into a fraction of the residual-stream norm."""
    if vector_dir is None:
        return None
    table = pair_table(vector_dir)
    scale: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (row["concept_pair"], row["layer"])
        if row["alpha"] != 0.0 and key not in scale:
            scale[key] = float(table.loc[row["concept_pair"], f"L{row['layer']:02d}_diff_norm"]) / residual_norm
    return scale


def antagonists(vector_dir: Path | None) -> dict[int, str]:
    if vector_dir is None:
        return {}
    table = pair_table(vector_dir)
    return dict(zip(table.index, table["antagonist"]))


def tidy(axis: plt.Axes) -> None:
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


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
                axis.barh(positions, values, height=height, color=colors[layer_index % len(colors)], label=f"слой {layer}")
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
            axis.set_title(f"α = {alpha:+g}")
            axis.grid(axis="x", alpha=0.25)
        axes[row][0].set_ylabel("к концепту" if sign > 0 else "к антагонисту")
    for axis in axes[-1]:
        axis.set_xlabel("Изменение accuracy\nотносительно α = 0, п.п.")
    axes[0][0].set_xlim(-1.45 * limit, 1.45 * limit)
    axes[0][0].set_yticks(range(len(concepts)), [textwrap.fill(names[pair], 26) for pair in concepts], fontsize=8)
    axes[0][0].set_ylim(len(concepts) - 0.5, -0.5)
    handles, labels = legend_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(layers), frameon=False)
    figure.suptitle(
        f"Влияние стиринга на accuracy ({full} вопросов; n подписано там, где посчитаны не все)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = out / "steering-accuracy.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def dose_response(
    rows: list[dict[str, Any]],
    baseline: dict[str, float],
    out: Path,
    scale: dict[tuple[int, int], float] | None = None,
    residual_norm: float = 92.0,
) -> Path:
    """Accuracy change and the reasoning length of right and wrong answers along alpha, per concept."""
    rng = np.random.default_rng(20260911)
    full = len(baseline)
    grouped = cells(rows)
    names = {row["concept_pair"]: row["concept"] for row in rows if row["alpha"] != 0.0}
    layers = sorted({layer for _, layer in grouped})
    base_rows = [row for row in rows if row["alpha"] == 0.0]

    def length(subset: list[dict[str, Any]], right: bool) -> float:
        values = [row["reasoning_token_count"] for row in subset if bool(row["correct"]) == right]
        # NaN breaks the line, so a dropped point shows as a gap rather than a straight join.
        return float(np.median(values)) if len(values) >= MIN_ANSWERS else float("nan")

    figure, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    dropped = {1: False, 2: False}
    partial_any = False
    for (pair, layer), by_alpha in sorted(grouped.items(), key=lambda item: names[item[0][0]]):
        color = concept_color(pair)
        points = [(0.0, 0.0, 0.0, 0.0, length(base_rows, True), length(base_rows, False), full)]
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
                    length(subset, True),
                    length(subset, False),
                    len(deltas),
                )
            )
        points.sort(key=lambda point: point[0])
        factor = scale.get((pair, layer), 1.0) if scale else 1.0
        x = [point[0] * factor for point in points]
        label = textwrap.shorten(names[pair], 42) + (f" (слой {layer})" if len(layers) > 1 else "")
        if scale:
            label += f"   |v| = {factor * residual_norm:.1f}"
        axes[0].plot(x, [point[1] for point in points], color=color, linewidth=2, marker="o", markersize=5, label=label)
        # The shaded band is where the mean would likely land on another draw of questions.
        axes[0].fill_between(
            x, [point[2] for point in points], [point[3] for point in points], color=color, alpha=0.12, linewidth=0
        )
        for panel, column in ((1, 4), (2, 5)):
            values = [point[column] for point in points]
            dropped[panel] = dropped[panel] or any(np.isnan(values))
            axes[panel].plot(x, values, color=color, linewidth=2, marker="o", markersize=5)
        # Hollow markers flag conditions that have not reached every question yet.
        partial = [point for point in points if point[6] < full]
        partial_any = partial_any or bool(partial)
        axes[0].scatter(
            [point[0] * factor for point in partial],
            [point[1] for point in partial],
            facecolors="white",
            edgecolors=color,
            zorder=3,
            s=42,
        )

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Accuracy", loc="left", fontsize=11)
    axes[0].set_ylabel("Изменение относительно\nα = 0, п.п.")
    gap = f" (точки, где таких ответов меньше {MIN_ANSWERS}, не показаны)"
    axes[1].set_title("Длина reasoning у верных ответов" + (gap if dropped[1] else ""), loc="left", fontsize=11)
    axes[2].set_title(
        "Длина reasoning у неверных ответов, включая незакрытые рассуждения" + (gap if dropped[2] else ""),
        loc="left",
        fontsize=11,
    )
    budget = max((row.get("max_new_tokens") or 0) for row in rows)
    for axis in axes[1:]:
        axis.set_ylabel("Медиана, токены")
        if budget:
            # Unfinished runs stop here, which is why wrong answers can sit on this line.
            axis.axhline(budget, color="#898781", linewidth=1, linestyle="--")
            axis.text(0.01, budget, f"лимит генерации {budget:,} токенов".replace(",", " "),
                      transform=axis.get_yaxis_transform(), va="bottom", fontsize=8, color="#52514e")
            axis.set_ylim(0, budget * 1.1)
        else:
            axis.set_ylim(bottom=0)
    axes[2].set_xlabel(
        "Сила стиринга, доля нормы residual stream\nминус — к антагонисту, плюс — к концепту"
        if scale
        else "α, во сколько раз прибавлен вектор концепта\nминус — к антагонисту, плюс — к концепту"
    )
    for axis in axes:
        tidy(axis)
        axis.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    accuracy = 100 * sum(baseline.values()) / full
    benchmarks = sorted({BENCHMARK_NAMES.get(row["benchmark"], row["benchmark"]) for row in rows})
    title = (
        "Отклик на стиринг" + (f", слой {layers[0]}" if len(layers) == 1 else "")
        + f"\n{', '.join(benchmarks)}: {full} вопросов, accuracy без стиринга {accuracy:.1f}%"
    )
    if partial_any:
        title += "; полые точки посчитаны не на всех вопросах"
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0.04 + 0.018 * ((len(handles) + 1) // 2), 1, 0.95))
    # The converted figure gets its own name, so both unit systems stay side by side.
    path = out / ("steering-dose-response-anthropic-units.png" if scale else "steering-dose-response.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def outcome_composition(
    rows: list[dict[str, Any]],
    full: int,
    out: Path,
    scale: dict[tuple[int, int], float] | None = None,
    opposite: dict[int, str] | None = None,
) -> Path:
    opposite = opposite or {}
    concepts = sorted({(row["concept_pair"], row["concept"]) for row in rows if row["concept_pair"]}, key=lambda item: item[1])
    alphas = sorted({row["alpha"] for row in rows})
    # Each concept converts alpha with its own factor, so each panel needs its own tick labels.
    figure, axes = plt.subplots(
        len(concepts), 1, figsize=(8, 2.4 + 2.2 * len(concepts)), sharex=not scale, squeeze=False
    )
    for row_index, (pair, name) in enumerate(concepts):
        axis = axes[row_index][0]
        layer = next(row["layer"] for row in rows if row["concept_pair"] == pair)
        factor = scale.get((pair, layer)) if scale else None
        labels, sizes = [], []
        counts = {key: [] for key in (CORRECT, WRONG, CUT)}
        for alpha in alphas:
            subset = [
                r for r in rows
                if (r["concept_pair"] == pair and r["alpha"] == alpha) or (alpha == 0.0 and r["alpha"] == 0.0)
            ]
            if not subset:
                continue
            labels.append(f"{alpha:g}" if factor is None or alpha == 0.0 else f"{alpha:g}\n{alpha * factor:+.2f}")
            sizes.append(len(subset))
            tally = collections.Counter(outcome(r) for r in subset)
            for key in counts:
                counts[key].append(100 * tally[key] / len(subset))
        bottoms = [0.0] * len(labels)
        for key in (CORRECT, WRONG, CUT):
            axis.bar(
                labels, counts[key], bottom=bottoms, color=OUTCOME_COLORS[key], label=key,
                width=0.6, edgecolor="white", linewidth=1,
            )
            bottoms = [b + v for b, v in zip(bottoms, counts[key])]
        for position, size in enumerate(sizes):
            if size < full:
                axis.text(position, 102, f"n={size}", ha="center", va="bottom", fontsize=7)
        axis.set_ylabel("% генераций")
        title = name + (f"  ↔  {opposite[pair]}" if pair in opposite else "")
        axis.set_title(textwrap.fill(title, 100), fontsize=10)
        axis.set_ylim(0, 112)
        axis.set_yticks(range(0, 101, 20))
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1][0].set_xlabel(
        "α (верхняя строка) и та же сила в долях нормы residual stream (нижняя строка); 0 — без стиринга"
        if scale
        else "α (0 — без стиринга)"
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Чем закончились генерации", fontsize=12)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = out / "steering-outcomes.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.results, args.benchmark, args.steering_version)
    if args.reasoning_scope == "closed":
        rows = [row for row in rows if row["reasoning_status"] == "closed_thinking"]
        if not rows:
            raise SystemExit("No closed-thinking records in the selected data")
    baseline = baseline_accuracy(rows)
    if not baseline:
        raise SystemExit("No alpha=0 baseline records; accuracy deltas need them")
    if not args.include_partial:
        rows = only_complete(rows, baseline)
    scale = strength_scale(args.vector_dir, args.residual_norm, rows)
    written = [
        accuracy_bars(rows, baseline, out),
        dose_response(rows, baseline, out, scale, args.residual_norm),
        outcome_composition(rows, len(baseline), out, scale, antagonists(args.vector_dir)),
    ]
    print(f"{len(rows)} records -> " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
