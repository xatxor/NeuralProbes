"""Compute token-level AdvBench mean/std separately for transcript regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = (11, 14, 18, 22, 25)
REGIONS = ("prompt", "assistant", "boundary", "response")
METHODS = ("raw", "projection")
PAIRS = 1036


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    shape = (len(REGIONS), len(LAYERS), PAIRS)
    sums = {method: np.zeros(shape, dtype=np.float64) for method in METHODS}
    squares = {method: np.zeros(shape, dtype=np.float64) for method in METHODS}
    counts = np.zeros((len(REGIONS), len(LAYERS)), dtype=np.int64)
    traces = sorted(args.results.glob("traces/*/*/meta.json"))
    for index, meta_path in enumerate(traces, 1):
        trace, regions = meta_path.parent, json.loads(meta_path.read_text())["regions"]
        for region_index, region in enumerate(REGIONS):
            start, end = regions[region]
            for layer_index, layer in enumerate(LAYERS):
                for method in METHODS:
                    values = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end].astype(np.float32)
                    sums[method][region_index, layer_index] += values.sum(axis=0, dtype=np.float64)
                    squares[method][region_index, layer_index] += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
                counts[region_index, layer_index] += end - start
        if index % 100 == 0:
            print(f"{index}/{len(traces)} traces", flush=True)
    output = {"regions": np.asarray(REGIONS), "layers": np.asarray(LAYERS), "counts": counts}
    for method in METHODS:
        mean = sums[method] / counts[..., None]
        variance = squares[method] / counts[..., None] - np.square(mean)
        output[f"{method}_mean"] = mean.astype(np.float32)
        output[f"{method}_std"] = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
    np.savez_compressed(args.results / "advbench_region_normalization.npz", **output)


if __name__ == "__main__":
    main()
