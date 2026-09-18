"""What steering one concept saves in reasoning length, and what it costs in accuracy.

Two paired comparisons against the unsteered baseline, question by question:

  - reasoning length over the questions both versions answered correctly, so the saving
    cannot come from runs that never finished their reasoning;
  - accuracy over every question.

Both carry bootstrap intervals over questions. Runs on the saved steering.jsonl.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_outcomes import benchmark_label, load, question_key  # noqa: E402

TOKENS_COLOR = "#2a78d6"
ACCURACY_COLOR = "#eb6834"
BOOTSTRAP_SAMPLES = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the figure (default: --results)")
    parser.add_argument(
        "--benchmark",
        default="gpqa_diamond",
        help="One benchmark, a comma-separated list to pool, or 'all'",
    )
    parser.add_argument("--concept-pair", type=int, default=657)
    parser.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated strengths; default: every positive strength present",
    )
    parser.add_argument("--steering-version", type=int, default=None)
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Measure token change over every paired generation, including unclosed thinking blocks. "
        "By default it is measured only where both answers are correct.",
    )
    return parser.parse_args()


def interval(values: np.ndarray, rng: np.random.Generator, log: bool = False) -> tuple[float, float, float]:
    """Point estimate and a 95% bootstrap interval over questions."""
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    point = float(values.mean())
    if log:
        return float(np.exp(point)), float(np.exp(low)), float(np.exp(high))
    return point, float(low), float(high)


def main() -> None:
    args = parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.results, args.benchmark, args.steering_version)
    baseline = {question_key(row): row for row in rows if row["alpha"] == 0.0}
    if not baseline:
        raise SystemExit("No alpha=0 baseline records; the comparison needs them")

    steered = [row for row in rows if row["concept_pair"] == args.concept_pair and row["alpha"] != 0.0]
    if not steered:
        raise SystemExit(f"No records for concept pair {args.concept_pair}")
    name = steered[0]["concept"]
    alphas = (
        [float(value) for value in args.alphas.split(",") if value.strip()]
        if args.alphas
        else sorted({row["alpha"] for row in steered if row["alpha"] > 0})
    )

    rng = np.random.default_rng(20260916)
    points = []
    for alpha in alphas:
        subset = [row for row in steered if row["alpha"] == alpha and question_key(row) in baseline]
        if not subset:
            continue
        token_rows = (
            subset
            if args.include_unfinished
            else [row for row in subset if row["correct"] and baseline[question_key(row)]["correct"]]
        )
        if len(token_rows) < 10:
            description = "paired generations" if args.include_unfinished else "questions answered correctly by both"
            print(f"alpha {alpha:+g}: only {len(token_rows)} {description}, skipped")
            continue
        ratio = np.log(
            np.array(
                [
                    row["reasoning_token_count"] / baseline[question_key(row)]["reasoning_token_count"]
                    for row in token_rows
                ]
            )
        )
        accuracy = np.array(
            [float(row["correct"]) - float(baseline[question_key(row)]["correct"]) for row in subset]
        )
        points.append(
            {
                "alpha": alpha,
                "tokens": interval(ratio, rng, log=True),
                "accuracy": interval(100 * accuracy, rng),
                "token_questions": len(token_rows),
                "questions": len(subset),
            }
        )
    if not points:
        raise SystemExit("Nothing to plot")

    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    x = np.arange(len(points))
    labels = [f"{point['alpha']:+g}" for point in points]

    # Left: the saving, as a percentage of the unsteered reasoning length.
    values = [100 * (point["tokens"][0] - 1) for point in points]
    errors = np.array([[100 * (point["tokens"][0] - point["tokens"][1]) for point in points],
                       [100 * (point["tokens"][2] - point["tokens"][0]) for point in points]])
    axes[0].bar(x, values, color=TOKENS_COLOR, width=0.6)
    axes[0].errorbar(x, values, yerr=errors, fmt="none", ecolor="#52514e", capsize=4, linewidth=1.2)
    for position, (value, point) in enumerate(zip(values, points)):
        axes[0].annotate(f"{value:+.0f}%", (position, value), textcoords="offset points",
                         xytext=(0, -14 if value < 0 else 6), ha="center", fontsize=10)
    axes[0].set_title("Длина рассуждения", loc="left", fontsize=12)
    axes[0].set_ylabel("Изменение числа токенов, %")

    # Right: what that costs in accuracy, over every question.
    values = [point["accuracy"][0] for point in points]
    errors = np.array([[point["accuracy"][0] - point["accuracy"][1] for point in points],
                       [point["accuracy"][2] - point["accuracy"][0] for point in points]])
    axes[1].bar(x, values, color=ACCURACY_COLOR, width=0.6)
    axes[1].errorbar(x, values, yerr=errors, fmt="none", ecolor="#52514e", capsize=4, linewidth=1.2)
    axes[1].set_title("Accuracy", loc="left", fontsize=12)
    axes[1].set_ylabel("Изменение, п.п.")

    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, labels)
        axis.set_xlabel("α, сила стиринга к концепту")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

    benchmark = benchmark_label(rows)
    token_questions = [point["token_questions"] for point in points]
    length_scope = (
        "по всем парным генерациям, включая незавершённые"
        if args.include_unfinished
        else "по вопросам, где обе версии ответили верно"
    )
    subtitle = (
        f"{benchmark}: длина — {length_scope} (n={min(token_questions)}–{max(token_questions)}); "
        f"accuracy — по всем {points[0]['questions']}"
    )
    figure.suptitle(f"{name}\n{textwrap.fill(subtitle, 115)}", fontsize=11, y=0.94)
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    path = out / f"tokens-vs-accuracy-pair-{args.concept_pair}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)

    for point in points:
        tokens, accuracy = point["tokens"], point["accuracy"]
        print(
            f"alpha {point['alpha']:+g}: tokens x{tokens[0]:.2f} [{tokens[1]:.2f}; {tokens[2]:.2f}] "
            f"on {point['token_questions']} questions, accuracy {accuracy[0]:+.1f} pp "
            f"[{accuracy[1]:+.1f}; {accuracy[2]:+.1f}] on {point['questions']}"
        )
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
