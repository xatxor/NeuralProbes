"""Collect response-level OpenThoughts math baselines without generating text."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibrate import LAYERS, MODEL_ID, load_vectors, model_input, shard_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_workers < 1 or args.chunk_size < 1:
        parser.error("--num-workers and --chunk-size must be positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    return args


def samples(manifests: list[Path]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for path in manifests for line in path.read_text().splitlines() if line]
    rows = [row for row in rows if row.get("domain") == "math"]
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 789 or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Expected 789 unique math responses, found {len(rows)}")
    return rows


class ResponseMeanCollector:
    def __init__(self, model: Any, vectors: dict[int, torch.Tensor]) -> None:
        self.model, self.vectors = model, vectors
        self.start = self.end = self.offset = 0
        self.count = np.zeros(len(LAYERS), dtype=np.int64)
        self.sums = np.zeros((len(LAYERS), 1036), dtype=np.float64)
        self.handles = [model.model.layers[layer - 1].register_forward_hook(self._hook(index, layer)) for index, layer in enumerate(LAYERS)]

    def _hook(self, layer_index: int, layer: int):
        def collect(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            lo, hi = max(self.start - self.offset, 0), min(self.end - self.offset, hidden.shape[1])
            if hi <= lo:
                return
            values = F.normalize(hidden[0, lo:hi].float(), dim=-1) @ self.vectors[layer].T
            values = values.to(torch.float16).cpu().numpy().astype(np.float64)
            self.sums[layer_index] += values.sum(axis=0)
            self.count[layer_index] += len(values)
        return collect

    @torch.inference_mode()
    def mean(self, token_ids: list[int], start: int, end: int, chunk_size: int) -> tuple[np.ndarray, int]:
        self.sums.fill(0)
        self.count.fill(0)
        self.start, self.end = start, end
        cache = None
        for self.offset in range(0, len(token_ids), chunk_size):
            chunk = torch.tensor(token_ids[self.offset : self.offset + chunk_size], device=self.model.device).unsqueeze(0)
            output = self.model.model(input_ids=chunk, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
        if not np.all(self.count == self.count[0]) or self.count[0] != end - start:
            raise RuntimeError("Layer hooks did not collect the complete reasoning span")
        return (self.sums / self.count[:, None]).astype(np.float32), int(self.count[0])

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Response baselines require CUDA")
    rows = samples(args.sample_manifests)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map={"": "cuda:0"}).eval()
    _, vectors = load_vectors(model.device)
    collector = ResponseMeanCollector(model, vectors)
    assigned = shard_indices(len(rows), args.worker_index, args.num_workers)
    means, ids, counts = [], [], []
    try:
        for progress, index in enumerate(assigned, 1):
            token_ids, start, end = model_input(tokenizer, rows[index])
            value, count = collector.mean(token_ids, start, end, args.chunk_size)
            means.append(value)
            ids.append(rows[index]["sample_id"])
            counts.append(count)
            print(f"[worker {args.worker_index}] {progress}/{len(assigned)} sample={rows[index]['sample_id']} reasoning={count}", flush=True)
    finally:
        collector.close()
    np.savez(args.output_dir / f"response_means.worker-{args.worker_index:02d}.npz", sample_ids=np.asarray(ids), means=np.asarray(means), reasoning_tokens=np.asarray(counts))


def visible_gpu_ids() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [item.strip() for item in value.split(",") if item.strip()] if value is not None else [str(index) for index in range(torch.cuda.device_count())]


def response_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def merge(args: argparse.Namespace) -> None:
    rows = samples(args.sample_manifests)
    parts = [np.load(args.output_dir / f"response_means.worker-{worker:02d}.npz") for worker in range(args.num_workers)]
    ids = np.concatenate([part["sample_ids"] for part in parts])
    means = np.concatenate([part["means"] for part in parts])
    counts = np.concatenate([part["reasoning_tokens"] for part in parts])
    order = np.argsort(ids)
    ids, means, counts = ids[order], means[order], counts[order]
    expected = np.asarray([row["sample_id"] for row in rows])
    if not np.array_equal(ids, expected):
        raise RuntimeError("Worker output IDs are incomplete or duplicated")
    mean, std = response_statistics(means)
    if not np.isfinite(std).all() or (std == 0).any():
        raise RuntimeError("Invalid response-level standard deviation")
    np.savez(args.output_dir / "response_means.npz", sample_ids=ids, means=means, reasoning_tokens=counts, layers=np.asarray(LAYERS))
    np.savez(args.output_dir / "response_normalization.npz", layers=np.asarray(LAYERS), pair_ids=np.arange(1036), count=len(ids), mean=mean, std=std)
    (args.output_dir / "metadata.json").write_text(json.dumps({"model": MODEL_ID, "method": "diff", "baseline_domain": "math", "responses": len(ids), "formula": "response mean cosine; baseline std across responses", "sample_ids": ids.tolist()[:3] + ["..."] + ids.tolist()[-3:]}, indent=2) + "\n")


def launch(args: argparse.Namespace) -> None:
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(f"Requested {args.num_workers} workers, visible GPUs: {gpu_ids}")
    processes = []
    for worker in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker]
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--sample-manifests", *map(str, args.sample_manifests), "--output-dir", str(args.output_dir), "--num-workers", str(args.num_workers), "--chunk-size", str(args.chunk_size), "--worker-index", str(worker)]
        processes.append(subprocess.Popen(command, env=environment))
    failures = [(worker, process.wait()) for worker, process in enumerate(processes) if process.wait()]
    if failures:
        raise RuntimeError(f"Workers failed: {failures}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        run_worker(args)
        return
    if args.num_workers == 1:
        args.worker_index = 0
        run_worker(args)
        args.worker_index = None
    else:
        launch(args)
    merge(args)
    print(f"Saved response baseline to {args.output_dir}")


if __name__ == "__main__":
    main()
