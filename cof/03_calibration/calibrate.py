"""Estimate token-level Qwen3 concept cosine statistics on ready OpenThoughts CoTs."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-8B"
VECTOR_REPO = "josephofthebread/Qwen3-8B-concept-vectors"
VECTOR_REVISION = "e15e1db9ca228c158aa4a372143922c8f66fb3c8"
LAYERS = (11, 14, 18, 22, 25)
DATASET_ID = "open-thoughts/OpenThoughts-114k"
DATASET_CONFIG = "metadata"
DEFAULT_OUTPUT = ROOT / "results"
METHOD = "diff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip this many valid seeded samples before selecting.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_samples < 1 or args.sample_offset < 0 or args.num_workers < 1 or args.chunk_size < 1:
        parser.error("--num-samples, --sample-offset, --num-workers, and --chunk-size must be valid")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    return args


def _find(sequence: list[int], needle: list[int], start: int = 0) -> int | None:
    for index in range(start, len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    return None


def clean_reasoning(text: str) -> str:
    text = text.strip()
    if text.startswith("<think>") and text.endswith("</think>"):
        return text[len("<think>") : -len("</think>")].strip()
    return text


def model_input(tokenizer: Any, sample: dict[str, Any]) -> tuple[list[int], int, int]:
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "<think>" not in prefix.rsplit("<|im_start|>assistant", 1)[-1]:
        prefix += "<think>\n"
    text = prefix + clean_reasoning(sample["reasoning"]) + "\n</think>\n\n" + sample["solution"]
    if tokenizer.eos_token:
        text += tokenizer.eos_token
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    start = _find(token_ids, opening)
    if start is None:
        raise ValueError("Qwen tokenizer could not locate <think>")
    start += len(opening)
    end = _find(token_ids, closing, start)
    if end is None or end <= start:
        raise ValueError("Qwen tokenizer could not locate a non-empty reasoning span")
    return token_ids, start, end


def load_excluded_problems() -> set[str]:
    return {
        row["problem"].strip()
        for row in load_dataset("HuggingFaceH4/aime_2024", split="train")
        if row.get("problem")
    }


def prepare_samples(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    max_tokens = AutoConfig.from_pretrained(MODEL_ID).max_position_embeddings
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    excluded = load_excluded_problems()
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    samples = []
    for dataset_index in indices:
        row = dataset[dataset_index]
        problem = row.get("problem")
        reasoning = row.get("deepseek_reasoning")
        solution = row.get("deepseek_solution") or row.get("ground_truth_solution")
        if not all(isinstance(value, str) and value.strip() for value in (problem, reasoning, solution)):
            continue
        if problem.strip() in excluded:
            continue
        sample = {
            "sample_id": len(samples),
            "dataset_index": dataset_index,
            "problem": problem,
            "reasoning": reasoning,
            "solution": solution,
            "domain": row.get("domain"),
            "source": row.get("source"),
        }
        token_ids, start, end = model_input(tokenizer, sample)
        if len(token_ids) > max_tokens:
            continue
        sample.update({"token_count": len(token_ids), "reasoning_token_count": end - start})
        samples.append(sample)
        if len(samples) == args.num_samples + args.sample_offset:
            break
    if len(samples) != args.num_samples + args.sample_offset:
        raise RuntimeError(f"Found only {len(samples)} valid samples out of {args.num_samples} requested")
    samples = samples[args.sample_offset :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def read_samples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def shard_indices(size: int, worker_index: int, num_workers: int) -> list[int]:
    return list(range(worker_index, size, num_workers))


def load_pairs() -> pd.DataFrame:
    return pd.read_parquet(
        hf_hub_download(VECTOR_REPO, "pairs.parquet", revision=VECTOR_REVISION)
    ).set_index("pair")


def load_vectors(device: torch.device) -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
    pairs = load_pairs()
    tensor = load_file(
        hf_hub_download(VECTOR_REPO, f"{METHOD}.safetensors", revision=VECTOR_REVISION)
    )[METHOD]
    if tuple(tensor.shape) != (len(LAYERS), 1036, 4096):
        raise ValueError(f"Unexpected {METHOD} vector shape: {tuple(tensor.shape)}")
    vectors = {
        layer: F.normalize(tensor[index].float(), dim=-1).to(device)
        for index, layer in enumerate(LAYERS)
    }
    return pairs, vectors


class CosineMoments:
    def __init__(self, model: Any, vectors: dict[int, torch.Tensor]) -> None:
        self.model = model
        self.vectors = vectors
        self.count = np.zeros(len(LAYERS), dtype=np.int64)
        self.sums = np.zeros((len(LAYERS), 1036), dtype=np.float64)
        self.sumsq = np.zeros_like(self.sums)
        self.offset = self.start = self.end = 0
        self.handles = [
            model.model.layers[layer - 1].register_forward_hook(self._hook(index, layer))
            for index, layer in enumerate(LAYERS)
        ]

    def _hook(self, layer_index: int, layer: int):
        def score(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            lo = max(self.start - self.offset, 0)
            hi = min(self.end - self.offset, hidden.shape[1])
            if hi <= lo:
                return
            values = F.normalize(hidden[0, lo:hi].float(), dim=-1) @ self.vectors[layer].T
            # Match the existing evaluation trace precision before calibration.
            values = values.to(torch.float16).cpu().numpy().astype(np.float64)
            self.count[layer_index] += len(values)
            self.sums[layer_index] += values.sum(axis=0)
            self.sumsq[layer_index] += np.square(values).sum(axis=0)

        return score

    @torch.inference_mode()
    def add(self, token_ids: list[int], start: int, end: int, chunk_size: int) -> None:
        cache = None
        self.start, self.end = start, end
        for self.offset in range(0, len(token_ids), chunk_size):
            chunk = torch.tensor(
                token_ids[self.offset : self.offset + chunk_size],
                dtype=torch.long,
                device=self.model.device,
            ).unsqueeze(0)
            output = self.model.model(input_ids=chunk, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
        del cache

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Calibration requires a CUDA GPU")
    samples = read_samples(args.output_dir / "samples.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": "cuda:0"}
    ).eval()
    _, vectors = load_vectors(model.device)
    moments = CosineMoments(model, vectors)
    started = time.perf_counter()
    assigned = shard_indices(len(samples), args.worker_index, args.num_workers)
    try:
        for progress, sample_index in enumerate(assigned, 1):
            token_ids, start, end = model_input(tokenizer, samples[sample_index])
            moments.add(token_ids, start, end, args.chunk_size)
            print(
                f"[worker {args.worker_index}] {progress}/{len(assigned)} "
                f"sample={sample_index} tokens={len(token_ids)} reasoning={end - start}",
                flush=True,
            )
    finally:
        moments.close()
    np.savez(
        args.output_dir / f"moments.worker-{args.worker_index:02d}.npz",
        count=moments.count,
        sum=moments.sums,
        sumsq=moments.sumsq,
        sample_count=len(assigned),
        elapsed_seconds=time.perf_counter() - started,
    )


def visible_gpu_ids() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [value.strip() for value in configured.split(",") if value.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def launch_workers(args: argparse.Namespace) -> None:
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(f"Requested {args.num_workers} workers, but only {gpu_ids} are visible")
    processes = []
    for worker_index in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_index]
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--num-samples",
            str(args.num_samples),
            "--sample-offset",
            str(args.sample_offset),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--chunk-size",
            str(args.chunk_size),
            "--output-dir",
            str(args.output_dir),
            "--worker-index",
            str(worker_index),
        ]
        print(f"Launching worker {worker_index} on GPU {gpu_ids[worker_index]}", flush=True)
        processes.append(subprocess.Popen(command, env=environment))
    failures = [
        (index, code)
        for index, code in enumerate(process.wait() for process in processes)
        if code
    ]
    if failures:
        raise RuntimeError(f"Calibration workers failed: {failures}")


def finish_statistics(count: np.ndarray, sums: np.ndarray, sumsq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expanded_count = np.broadcast_to(count[:, None], sums.shape)
    mean = sums / expanded_count
    variance = np.maximum((sumsq - np.square(sums) / expanded_count) / (expanded_count - 1), 0)
    return mean, variance, np.sqrt(variance)


def merge_results(args: argparse.Namespace, wall_seconds: float) -> None:
    parts = [
        np.load(args.output_dir / f"moments.worker-{worker:02d}.npz")
        for worker in range(args.num_workers)
    ]
    count = sum((part["count"] for part in parts), start=np.zeros(len(LAYERS), dtype=np.int64))
    sums = sum((part["sum"] for part in parts), start=np.zeros((len(LAYERS), 1036)))
    sumsq = sum((part["sumsq"] for part in parts), start=np.zeros((len(LAYERS), 1036)))
    if np.any(count < 2):
        raise RuntimeError("At least two reasoning tokens are required for variance")
    mean, variance, std = finish_statistics(count, sums, sumsq)
    expanded_count = np.broadcast_to(count[:, None], mean.shape).copy()
    np.savez(
        args.output_dir / "normalization.npz",
        layers=np.asarray(LAYERS),
        pair_ids=np.arange(1036),
        count=expanded_count,
        mean=mean,
        variance=variance,
        std=std,
    )
    pairs = load_pairs()
    rows = []
    for layer_index, layer in enumerate(LAYERS):
        for pair in range(1036):
            metadata = pairs.loc[pair]
            rows.append(
                {
                    "method": METHOD,
                    "layer": layer,
                    "pair": pair,
                    "concept": metadata["concept"],
                    "antagonist": metadata["antagonist"],
                    "token_count": int(count[layer_index]),
                    "mean_cosine": mean[layer_index, pair],
                    "variance_cosine": variance[layer_index, pair],
                    "std_cosine": std[layer_index, pair],
                }
            )
    pd.DataFrame(rows).to_parquet(args.output_dir / "concept_stats.parquet", index=False)
    samples = read_samples(args.output_dir / "samples.jsonl")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "model": MODEL_ID,
        "vector_repo": VECTOR_REPO,
        "vector_revision": VECTOR_REVISION,
        "method": METHOD,
        "layers": list(LAYERS),
        "num_samples": len(samples),
        "seed": args.seed,
        "sample_offset": args.sample_offset,
        "chunk_size": args.chunk_size,
        "num_workers": args.num_workers,
        "cosine_trace_dtype": "float16",
        "reasoning_tokens": int(count[0]),
        "wall_seconds": wall_seconds,
        "worker_seconds": [float(part["elapsed_seconds"]) for part in parts],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.worker_index is not None:
        run_worker(args)
        return
    started = time.perf_counter()
    prepare_samples(args)
    if args.num_workers == 1:
        args.worker_index = 0
        run_worker(args)
        args.worker_index = None
    else:
        launch_workers(args)
    merge_results(args, time.perf_counter() - started)
    print(f"Saved calibration to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
