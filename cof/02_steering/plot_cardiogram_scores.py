"""Draw cardiograms from a saved score archive, without a GPU or model replay."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_concepts import score_benchmark_label  # noqa: E402
from plot_outcomes import pair_table  # noqa: E402
from steer import DEFAULT_CONCEPT_PAIRS  # noqa: E402

BOOTSTRAP_SAMPLES = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--concept-pairs",
        default=None,
        help="Comma-separated pair IDs; default: the 16 CoT concepts plus joy.",
    )
    parser.add_argument("--label", default=None, help="Optional suffix added to the PNG file name")
    return parser.parse_args()


def bootstrap_band(stacks: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap within benchmark, then give each benchmark equal weight."""
    draws = np.empty((BOOTSTRAP_SAMPLES, stacks[0].shape[1]), dtype=np.float32)
    batch = 100
    for start in range(0, BOOTSTRAP_SAMPLES, batch):
        stop = min(start + batch, BOOTSTRAP_SAMPLES)
        batch_draws = []
        for stack in stacks:
            indices = rng.integers(0, len(stack), size=(stop - start, len(stack)))
            batch_draws.append(stack[indices].mean(axis=1))
        draws[start:stop] = np.mean(batch_draws, axis=0)
    return tuple(np.quantile(draws, [0.025, 0.975], axis=0))


def main() -> None:
    args = parse_args()
    out = args.out or args.scores.parent
    out.mkdir(parents=True, exist_ok=True)
    pairs = (
        [int(value) for value in args.concept_pairs.split(",") if value.strip()]
        if args.concept_pairs
        else list(DEFAULT_CONCEPT_PAIRS)
    )

    with np.load(args.scores, allow_pickle=False) as data:
        pair_ids = data["pair_ids"].astype(int)
        correct = data["correct"].astype(bool)
        curves = data["binned_cosine"].astype(np.float32)
        token_mean = data["token_mean"].astype(np.float32)
        token_std = data["token_std"].astype(np.float32)
        bins = curves.shape[2]
        benchmark = score_benchmark_label(data)
        layer = int(data["layer"])
        alpha = float(data["alpha"])
        statuses = data["reasoning_status"].astype(str) if "reasoning_status" in data else None
        trace_benchmarks = (
            data["trace_benchmarks"].astype(str)
            if "trace_benchmarks" in data
            else np.full(len(correct), benchmark)
        )
        standardized = data["binned_z"].astype(np.float32) if "binned_z" in data else None

    if correct.sum() < 2 or (~correct).sum() < 2:
        raise SystemExit("Both groups need at least two traces")
    position = {pair: index for index, pair in enumerate(pair_ids)}
    missing = sorted(set(pairs) - set(position))
    if missing:
        raise SystemExit(f"Pairs not present in {args.scores}: {missing}")
    table = pair_table(args.vector_dir)
    if standardized is None:
        standardized = (curves - token_mean[None, :, None]) / token_std[None, :, None]

    rng = np.random.default_rng(20260916)
    x = (np.arange(bins) + 0.5) * 100 / bins
    columns = 2 if len(pairs) > 1 else 1
    rows = (len(pairs) + columns - 1) // columns
    extra_height = 0.8 if rows == 1 else 0.0
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.5 * columns, 3.0 * rows + extra_height),
        squeeze=False,
        sharex=True,
    )
    for index, pair in enumerate(pairs):
        axis = axes[index // columns][index % columns]
        concept_index = position[pair]
        for mask, label, color in (
            (correct, "Correct", "#3b76af"),
            (~correct, "Incorrect", "#d95f4c"),
        ):
            stacks = [
                standardized[mask & (trace_benchmarks == name), concept_index]
                for name in sorted(set(trace_benchmarks))
                if np.any(mask & (trace_benchmarks == name))
            ]
            count = sum(len(stack) for stack in stacks)
            mean = np.mean([stack.mean(axis=0) for stack in stacks], axis=0)
            axis.plot(x, mean, label=f"{label} (n={count})", color=color, linewidth=1.6)
            low, high = bootstrap_band(stacks, rng)
            axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(textwrap.fill(str(table.loc[pair, "concept"]), 44), fontsize=10)
        axis.set_ylabel("Signed z-mean")
        axis.grid(alpha=0.25)
    for index in range(len(pairs), rows * columns):
        axes[index // columns][index % columns].remove()
    for column in range(columns):
        if axes[rows - 1][column] in figure.axes:
            axes[rows - 1][column].set_xlabel("Relative position within reasoning, %")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    scope = ""
    if statuses is not None:
        unfinished = int(np.sum(statuses != "closed_thinking"))
        scope = f"; including {unfinished} unfinished" if unfinished else "; completed reasoning only"
    balance = "; benchmark-balanced" if len(set(trace_benchmarks)) > 1 else ""
    heading = (
        f"Concept activation along the reasoning trace: {benchmark} "
        f"(L{layer}, alpha={alpha:g}{scope}{balance})"
    )
    figure.suptitle("\n".join(textwrap.wrap(heading, 65)), y=0.98, va="top")
    top = 0.86 if rows == 1 else 0.92
    figure.tight_layout(rect=(0, 0.05, 1, top))
    suffix = f"-{args.label}" if args.label else ""
    path = out / f"cardiogram-L{layer}-a{alpha:g}{suffix}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(
        f"{len(correct)} traces ({int(correct.sum())} correct / {int((~correct).sum())} incorrect) -> {path}"
    )


if __name__ == "__main__":
    main()
