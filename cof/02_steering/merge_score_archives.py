"""Relabel and pool saved cardiogram score archives without replaying the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plot_outcomes import load, question_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="BENCHMARK=NPZ",
        help="A closed-trace score archive; repeat to pool benchmarks.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steering-version", type=int, default=None)
    return parser.parse_args()


def source_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be BENCHMARK=NPZ")
    benchmark, raw_path = value.split("=", 1)
    if not benchmark or not raw_path:
        raise argparse.ArgumentTypeError("--source must be BENCHMARK=NPZ")
    return benchmark, Path(raw_path)


def main() -> None:
    args = parse_args()
    sources = [source_spec(value) for value in args.source]
    selected = ",".join(benchmark for benchmark, _ in sources)
    rows = load(args.results, selected, args.steering_version)
    baseline = {
        question_key(row): row
        for row in rows
        if row["alpha"] == 0.0
    }

    pair_ids: np.ndarray | None = None
    ids: list[np.ndarray] = []
    trace_benchmarks: list[np.ndarray] = []
    statuses: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    reasoning_tokens: list[np.ndarray] = []
    means: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    standardized_curves: list[np.ndarray] = []
    token_total = 0
    token_sum: np.ndarray | None = None
    token_square: np.ndarray | None = None
    layer: int | None = None
    alpha: float | None = None

    for benchmark, path in sources:
        with np.load(path, allow_pickle=False) as data:
            candidate_pairs = data["pair_ids"].astype(np.int32)
            if pair_ids is None:
                pair_ids = candidate_pairs
                token_sum = np.zeros(len(pair_ids), dtype=np.float64)
                token_square = np.zeros(len(pair_ids), dtype=np.float64)
                layer = int(data["layer"])
                alpha = float(data["alpha"])
            elif not np.array_equal(pair_ids, candidate_pairs):
                raise SystemExit(f"Concept pairs differ in {path}")
            if int(data["layer"]) != layer or float(data["alpha"]) != alpha:
                raise SystemExit(f"Layer/alpha differs in {path}")

            source_ids = data["ids"].astype(str)
            source_correct = []
            source_status = []
            for example_id in source_ids:
                row = baseline.get((benchmark, str(example_id)))
                if row is None:
                    raise SystemExit(f"No baseline JSONL record for {benchmark}:{example_id}")
                source_correct.append(bool(row["correct"]))
                source_status.append(row["reasoning_status"])

            counts = data["reasoning_tokens"].astype(np.int64)
            count = int(counts.sum())
            source_mean = data["token_mean"].astype(np.float64)
            source_std = data["token_std"].astype(np.float64)
            assert token_sum is not None and token_square is not None
            token_sum += count * source_mean
            token_square += count * (np.square(source_std) + np.square(source_mean))
            token_total += count

            ids.append(source_ids)
            trace_benchmarks.append(np.full(len(source_ids), benchmark))
            statuses.append(np.asarray(source_status))
            correct.append(np.asarray(source_correct, dtype=bool))
            reasoning_tokens.append(counts.astype(np.int32))
            means.append(data["mean_cosine"].astype(np.float32))
            source_curves = data["binned_cosine"].astype(np.float32)
            curves.append(source_curves.astype(np.float16))
            standardized_curves.append(
                ((source_curves - source_mean[None, :, None]) / source_std[None, :, None]).astype(np.float16)
            )
            print(
                f"{benchmark}: {len(source_ids)} closed traces, "
                f"{sum(source_correct)} correct / {len(source_ids) - sum(source_correct)} incorrect"
            )

    assert pair_ids is not None and token_sum is not None and token_square is not None
    pooled_mean = token_sum / token_total
    pooled_std = np.sqrt(np.maximum(token_square / token_total - np.square(pooled_mean), 0.0))
    pooled_std[pooled_std == 0] = 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pair_ids=pair_ids,
        ids=np.concatenate(ids),
        trace_benchmarks=np.concatenate(trace_benchmarks),
        reasoning_status=np.concatenate(statuses),
        correct=np.concatenate(correct),
        reasoning_tokens=np.concatenate(reasoning_tokens),
        mean_cosine=np.concatenate(means, axis=0),
        binned_cosine=np.concatenate(curves, axis=0),
        binned_z=np.concatenate(standardized_curves, axis=0),
        token_mean=pooled_mean.astype(np.float32),
        token_std=pooled_std.astype(np.float32),
        benchmark=",".join(benchmark for benchmark, _ in sources),
        benchmarks=np.asarray([benchmark for benchmark, _ in sources]),
        layer=layer,
        alpha=alpha,
    )
    print(f"{sum(len(item) for item in ids)} traces -> {args.out}")


if __name__ == "__main__":
    main()
