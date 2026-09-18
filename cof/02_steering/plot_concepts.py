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
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_outcomes import BENCHMARK_NAMES, pair_table  # noqa: E402

CORRECT_COLOR = "#2a78d6"
INCORRECT_COLOR = "#eb6834"
CONNECTOR_COLOR = "#d6d3cc"
BOOTSTRAP_SAMPLES = 2_000
PERMUTATIONS = 1_000
# joy is the control of the grid: it belongs in the cardiogram, not among the CoT concepts.
CONTROL_PAIRS = (532,)


def resolve_pairs(value: str) -> set[int] | None:
    """Which concepts compete for the top rows: all of them, the CoT grid, or explicit ids."""
    if value == "all":
        return None
    if value == "cot":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from steer import DEFAULT_CONCEPT_PAIRS

        return {pair for pair in DEFAULT_CONCEPT_PAIRS if pair not in CONTROL_PAIRS}
    return {int(item) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="concept-scores-*.npz from cardiogram.py")
    parser.add_argument("--vector-dir", type=Path, required=True, help="For the concept and antagonist names")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the figure (default: next to --scores)")
    parser.add_argument("--top", type=int, default=8, help="How many concepts to show")
    parser.add_argument(
        "--pairs",
        default="all",
        help="Which concepts compete: all, cot (the 16 CoT concepts of the grid), or ids",
    )
    parser.add_argument("--label", default=None, help="Suffix for the file name; defaults to --pairs")
    parser.add_argument(
        "--title",
        default="Какие фичи активируются на правильном/неправильном CoT?",
        help="Figure title",
    )
    return parser.parse_args()


def score_benchmark_label(data: np.lib.npyio.NpzFile) -> str:
    """Describe the benchmark provenance stored by old or new cardiogram runs."""
    if "benchmarks" in data:
        names = [str(value) for value in data["benchmarks"]]
    elif "benchmark" in data:
        names = [item.strip() for item in str(data["benchmark"].item()).split(",") if item.strip()]
    else:
        return ""
    return " + ".join(BENCHMARK_NAMES.get(name, name) for name in names)


def main() -> None:
    args = parse_args()
    out = args.out or args.scores.parent
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.scores, allow_pickle=False) as data:
        benchmark = score_benchmark_label(data)
        pair_ids = data["pair_ids"]
        correct = data["correct"].astype(bool)
        raw = data["mean_cosine"].astype(np.float32)
        trace_benchmarks = (
            data["trace_benchmarks"].astype(str)
            if "trace_benchmarks" in data
            else np.full(len(correct), benchmark)
        )
    if correct.sum() < 2 or (~correct).sum() < 2:
        raise SystemExit("Both groups need at least two traces")

    # The pool is narrowed before anything is measured, so the chance level below is the one
    # for the concepts that actually compete.
    wanted = resolve_pairs(args.pairs)
    if wanted is not None:
        keep = [index for index, pair in enumerate(pair_ids) if int(pair) in wanted]
        if len(keep) < 2:
            raise SystemExit(f"Only {len(keep)} of the chosen concepts are in {args.scores}")
        pair_ids, raw = pair_ids[keep], raw[:, keep]

    # Standardise within benchmark.  Pooling first would let the very different GPQA and
    # MATH accuracies turn benchmark identity into an apparent correctness feature.
    benchmark_masks = {
        name: trace_benchmarks == name
        for name in sorted(set(trace_benchmarks))
    }
    usable = {
        name: mask
        for name, mask in benchmark_masks.items()
        if np.sum(mask & correct) >= 2 and np.sum(mask & ~correct) >= 2
    }
    if not usable:
        raise SystemExit("No benchmark has at least two correct and two incorrect traces")
    z = np.empty_like(raw)
    for mask in benchmark_masks.values():
        std = raw[mask].std(axis=0)
        std[std == 0] = 1.0
        z[mask] = (raw[mask] - raw[mask].mean(axis=0)) / std
    right_mean = np.mean([z[mask & correct].mean(axis=0) for mask in usable.values()], axis=0)
    wrong_mean = np.mean([z[mask & ~correct].mean(axis=0) for mask in usable.values()], axis=0)
    gap = right_mean - wrong_mean

    # Largest separation first, positive (correct higher) above negative, as in the report.
    order = np.argsort(-np.abs(gap))[: args.top]
    order = order[np.argsort(-gap[order])]

    rng = np.random.default_rng(20260916)
    draws = np.empty((BOOTSTRAP_SAMPLES, z.shape[1]), dtype=np.float32)
    batch = 20
    for start in range(0, BOOTSTRAP_SAMPLES, batch):
        stop = min(start + batch, BOOTSTRAP_SAMPLES)
        benchmark_draws = []
        for mask in usable.values():
            right = z[mask & correct]
            wrong = z[mask & ~correct]
            sampled_right = right[rng.integers(0, len(right), size=(stop - start, len(right)))].mean(axis=1)
            sampled_wrong = wrong[rng.integers(0, len(wrong), size=(stop - start, len(wrong)))].mean(axis=1)
            benchmark_draws.append(sampled_right - sampled_wrong)
        draws[start:stop] = np.mean(benchmark_draws, axis=0)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)

    # Over a thousand concepts the largest gap is large by chance alone, so the chance level
    # is measured rather than assumed: the correct labels are shuffled and the largest gap
    # any concept reaches is kept. A concept means something only above that line.
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        benchmark_spreads = []
        for mask in usable.values():
            values = z[mask]
            right_count = int(np.sum(mask & correct))
            shuffled = rng.permutation(len(values))
            benchmark_spreads.append(
                values[shuffled[:right_count]].mean(axis=0)
                - values[shuffled[right_count:]].mean(axis=0)
            )
        spread = np.mean(benchmark_spreads, axis=0)
        null[index] = np.abs(spread).max()
    threshold = float(np.quantile(null, 0.95))

    table = pair_table(args.vector_dir)
    labels, rows = [], []
    for index in order:
        pair = int(pair_ids[index])
        concept, antagonist = table.loc[pair, "concept"], table.loc[pair, "antagonist"]
        labels.append(f"{concept}\n↔ {antagonist}")
        rows.append((pair, concept, antagonist, gap[index], low[index], high[index]))

    figure, axis = plt.subplots(figsize=(11, 1.0 + 0.72 * len(order)))
    positions = np.arange(len(order))
    axis.hlines(positions, wrong_mean[order], right_mean[order],
                color=CONNECTOR_COLOR, linewidth=4, zorder=1)
    axis.scatter(right_mean[order], positions, color=CORRECT_COLOR, s=70, zorder=2, label="Correct")
    axis.scatter(wrong_mean[order], positions, color=INCORRECT_COLOR, s=70, zorder=2, label="Incorrect")
    axis.axvline(0, color="#52514e", linewidth=1)
    axis.set_yticks(positions, labels, fontsize=9)
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.set_xlabel(
        "Signed z-mean\n"
        f"перестановка меток по {len(pair_ids)} концептам даёт разрыв до {threshold:.2f}: "
        "меньше этого — неотличимо от случайности"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    balance = " (benchmark-balanced)" if len(usable) > 1 else ""
    title = f"{args.title}\n{benchmark}{balance}" if benchmark else args.title
    axis.set_title(title, fontsize=15, pad=16)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    figure.tight_layout()
    label = args.label if args.label is not None else ("" if args.pairs == "all" else args.pairs)
    path = out / f"concepts-correct-vs-incorrect{'-' + label.replace(',', '-') if label else ''}.png"
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
