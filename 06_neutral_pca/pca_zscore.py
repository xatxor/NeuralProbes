"""Estimate per-concept neutral-PCA projection mean and standard deviation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


PIPELINE = module("pca_zscore_pipeline", ROOT / "pipeline.py")
PROJECTION = PIPELINE.PROJECTION
LAYERS = PROJECTION.LAYERS
PAIRS = PROJECTION.PAIRS


def worker(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    probes = PROJECTION.load_probe_vectors(args.background, device)
    records = []
    for layer in LAYERS[args.worker :: args.workers]:
        total = torch.zeros(PAIRS, device=device)
        squares = torch.zeros(PAIRS, device=device)
        count = 0
        for source_worker in range(args.workers):
            values = np.load(args.background / f"background.worker-{source_worker:02d}-L{layer}.npy", mmap_mode="r")
            for start in range(0, len(values), args.batch_size):
                hidden = torch.from_numpy(values[start : start + args.batch_size]).to(device=device, dtype=torch.float16)
                scores = hidden.float() @ probes[layer].T
                total += scores.sum(dim=0)
                squares += scores.square().sum(dim=0)
                count += len(scores)
        mean = total / count
        variance = (squares / count - mean.square()).clamp_min(0)
        std = variance.sqrt().clamp_min(1e-6)
        records.append((layer, mean.cpu().numpy(), std.cpu().numpy(), count))
        print(f"L{layer}: {count} neutral tokens", flush=True)
    np.savez(
        args.output / f"pca_zscore.worker-{args.worker:02d}.npz",
        layers=np.asarray([row[0] for row in records]),
        mean=np.asarray([row[1] for row in records]),
        std=np.asarray([row[2] for row in records]),
        count=np.asarray([row[3] for row in records]),
    )


def launch(args: argparse.Namespace) -> None:
    jobs = []
    for worker_index in range(args.workers):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        jobs.append(subprocess.Popen([
            sys.executable, "-u", __file__, "--background", str(args.background),
            "--output", str(args.output), "--workers", str(args.workers),
            "--worker", str(worker_index), "--batch-size", str(args.batch_size),
        ], env=env))
    failures = [(index, job.wait()) for index, job in enumerate(jobs)]
    if any(code for _, code in failures):
        raise RuntimeError(f"PCA z-score workers failed: {failures}")
    mean, std, count = {}, {}, {}
    for worker_index in range(args.workers):
        values = np.load(args.output / f"pca_zscore.worker-{worker_index:02d}.npz")
        for layer, layer_mean, layer_std, layer_count in zip(values["layers"], values["mean"], values["std"], values["count"], strict=True):
            mean[int(layer)], std[int(layer)], count[int(layer)] = layer_mean, layer_std, int(layer_count)
    if set(mean) != set(LAYERS):
        raise RuntimeError("Missing PCA z-score layer")
    np.savez(args.output / "pca_projection_normalization.npz", mean=np.stack([mean[layer] for layer in LAYERS]), std=np.stack([std[layer] for layer in LAYERS]))
    (args.output / "pca_projection_normalization.json").write_text(json.dumps({"layers": list(LAYERS), "tokens_per_layer": count, "score": "PCA-projected signed residual dot product"}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker(args) if args.worker is not None else launch(args)


if __name__ == "__main__":
    main()
