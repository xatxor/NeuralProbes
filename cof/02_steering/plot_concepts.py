"""Which concepts separate correct reasoning from incorrect, as a dumbbell chart.

The method is the one the previous phase used (vika/01_eval):

  1. per trace and concept, the mean cosine between the residual stream and the concept
     direction over the reasoning span (concept_analysis.py);
  2. those per-trace means standardised within the concept, across traces, which is the
     signed z-mean (build_concept_report.py);
  3. the correct and the incorrect traces averaged apart, and concepts ranked by the gap
     between the two groups.

Reads the scores saved by cardiogram.py, so it needs neither a GPU nor the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CORRECT_COLOR = "#2a78d6"
INCORRECT_COLOR = "#eb6834"
CONNECTOR_COLOR = "#d6d3cc"
BOOTSTRAP_SAMPLES = 2_000
PERMUTATIONS = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="concept-scores-*.npz from cardiogram.py")
    parser.add_argument("--vector-dir", type=Path, required=True, help="For the concept and antagonist names")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the figure (default: next to --scores)")
    parser.add_argument("--top", type=int, default=8, help="How many concepts to show")
    parser.add_argument(
        "--title",
        default="Какие фичи активируются на правильном/неправильном CoT?",
        help="Figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.scores.parent
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.scores, allow_pickle=False)
    pair_ids = data["pair_ids"]
    correct = data["correct"].astype(bool)
    raw = data["mean_cosine"].astype(np.float64)
    if correct.sum() < 2 or (~correct).sum() < 2:
        raise SystemExit("Both groups need at least two traces")

    # Standardise each concept across traces, as the previous phase did before averaging.
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    z = (raw - raw.mean(axis=0)) / std
    right, wrong = z[correct], z[~correct]
    gap = right.mean(axis=0) - wrong.mean(axis=0)

    # Largest separation first, positive (correct higher) above negative, as in the report.
    order = np.argsort(-np.abs(gap))[: args.top]
    order = order[np.argsort(-gap[order])]

    rng = np.random.default_rng(20260916)
    draws = (
        right[rng.integers(0, len(right), size=(BOOTSTRAP_SAMPLES, len(right)))].mean(axis=1)
        - wrong[rng.integers(0, len(wrong), size=(BOOTSTRAP_SAMPLES, len(wrong)))].mean(axis=1)
    )
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)

    # Over a thousand concepts the largest gap is large by chance alone, so the chance level
    # is measured rather than assumed: the correct labels are shuffled and the largest gap
    # any concept reaches is kept. A concept means something only above that line.
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        shuffled = rng.permutation(len(z))
        spread = z[shuffled[: len(right)]].mean(axis=0) - z[shuffled[len(right) :]].mean(axis=0)
        null[index] = np.abs(spread).max()
    threshold = float(np.quantile(null, 0.95))

    table = pd.read_parquet(args.vector_dir / "pairs.parquet").set_index("pair_id")
    labels, rows = [], []
    for index in order:
        pair = int(pair_ids[index])
        concept, antagonist = table.loc[pair, "concept"], table.loc[pair, "antagonist"]
        labels.append(f"{concept}\n↔ {antagonist}")
        rows.append((pair, concept, antagonist, gap[index], low[index], high[index]))

    figure, axis = plt.subplots(figsize=(11, 1.0 + 0.72 * len(order)))
    positions = np.arange(len(order))
    axis.hlines(positions, wrong[:, order].mean(axis=0), right[:, order].mean(axis=0),
                color=CONNECTOR_COLOR, linewidth=4, zorder=1)
    axis.scatter(right[:, order].mean(axis=0), positions, color=CORRECT_COLOR, s=70, zorder=2, label="Correct")
    axis.scatter(wrong[:, order].mean(axis=0), positions, color=INCORRECT_COLOR, s=70, zorder=2, label="Incorrect")
    axis.axvline(0, color="#52514e", linewidth=1)
    axis.set_yticks(positions, labels, fontsize=9)
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.set_xlabel(
        "Signed z-mean\n"
        f"перестановка меток даёт разрыв до {threshold:.2f}: меньше этого — неотличимо от случайности"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_title(args.title, fontsize=15, pad=16)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    figure.tight_layout()
    path = out / "concepts-correct-vs-incorrect.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    # The interval says which rows would survive another sample of questions.
    print(f"correct traces: {int(correct.sum())}, incorrect: {int((~correct).sum())}")
    print(f"shuffled labels reach a gap of {threshold:.3f}; only larger gaps mean anything")
    print(f"{'pair':>6}  {'gap':>6}  {'95% interval':>16}  concept")
    passed = 0
    for pair, concept, antagonist, value, lo, hi in rows:
        mark = "*" if abs(value) > threshold else " "
        passed += mark == "*"
        print(f"{pair:>6}  {value:+.3f}  [{lo:+.3f}; {hi:+.3f}]{mark} {concept} vs {antagonist}")
    print(f"{passed} of {len(rows)} above the shuffled-label level. Figure -> {path}")


if __name__ == "__main__":
    main()
