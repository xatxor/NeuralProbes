"""Fit background PCs and score saved AIME outputs with denoised linear probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LAYERS = (11, 14, 18, 22, 25)
MODEL_ID = "Qwen/Qwen3-8B"
VECTOR_REPO = "josephofthebread/Qwen3-8B-concept-vectors"
VECTOR_REVISION = "e15e1db9ca228c158aa4a372143922c8f66fb3c8"
PAIRS = 1036


def existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exist: {paths}")


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL_DIR = existing(PROJECT / "01_eval", PROJECT / "vika" / "01_eval")
CALIBRATION_DIR = existing(
    PROJECT / "03_calibration", PROJECT / "oleg" / "03_calibration"
)
sys.path.insert(0, str(EVAL_DIR))
CALIBRATE = load_local_module(
    "projection_calibrate", CALIBRATION_DIR / "calibrate.py"
)
EVALUATE = load_local_module("projection_evaluate", EVAL_DIR / "evaluate.py")


def visible_gpu_ids() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [value.strip() for value in configured.split(",") if value.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def allocate_counts(populations: list[int], total: int) -> list[int]:
    """Allocate an exact sample count proportionally across worker token pools."""
    if total < 1 or sum(populations) < total:
        raise ValueError("PCA sample count must be positive and no larger than the token population")
    shares = np.asarray(populations, dtype=np.float64) * total / sum(populations)
    counts = np.floor(shares).astype(int)
    for index in np.argsort(-(shares - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return counts.tolist()


def find_sequence(tokens: list[int], needle: list[int], start: int = 0) -> int | None:
    for index in range(start, len(tokens) - len(needle) + 1):
        if tokens[index : index + len(needle)] == needle:
            return index
    return None


def thinking_span(tokenizer: Any, output_ids: list[int]) -> tuple[int, int]:
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    start = find_sequence(output_ids, opening)
    if start is None:
        raise ValueError("Saved output has no <think> tag")
    start += len(opening)
    end = find_sequence(output_ids, closing, start)
    if end is None:
        end = len(output_ids)
    if end <= start:
        raise ValueError("Saved output has an empty reasoning span")
    return start, end


def load_model(tokenizer: Any | None = None) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": "cuda:0"}
    ).eval()
    return model, tokenizer


def run_chunks(model: Any, token_ids: list[int], collector: Any, chunk_size: int) -> None:
    cache = None
    with torch.inference_mode():
        for collector.offset in range(0, len(token_ids), chunk_size):
            chunk = torch.tensor(
                token_ids[collector.offset : collector.offset + chunk_size],
                dtype=torch.long,
                device=model.device,
            ).unsqueeze(0)
            result = model.model(input_ids=chunk, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
    del cache


class ActivationSampler:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.offset = 0
        self.selected = np.empty(0, dtype=np.int64)
        self.values: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
        self.handles = [
            model.model.layers[layer - 1].register_forward_hook(self._hook(layer))
            for layer in LAYERS
        ]

    def _hook(self, layer: int):
        def collect(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            selected = self.selected[
                (self.selected >= self.offset)
                & (self.selected < self.offset + hidden.shape[1])
            ]
            if len(selected):
                positions = torch.as_tensor(
                    selected - self.offset, dtype=torch.long, device=hidden.device
                )
                self.values[layer].append(
                    hidden[0].index_select(0, positions).to(torch.float16).cpu().numpy()
                )

        return collect

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def read_background_samples(manifests: list[Path]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for path in manifests
        for line in path.read_text().splitlines()
        if line
    ]
    rows = [row for row in rows if row.get("domain") == "math"]
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 789 or len({row["sample_id"] for row in rows}) != 789:
        raise RuntimeError(f"Expected 789 unique math responses, found {len(rows)}")
    return rows


def background_worker(args: argparse.Namespace) -> None:
    rows = read_background_samples(args.sample_manifests)
    shards = [rows[index :: args.num_workers] for index in range(args.num_workers)]
    populations = [
        sum(int(row["reasoning_token_count"]) for row in shard) for shard in shards
    ]
    target = allocate_counts(populations, args.pca_samples)[args.worker_index]
    assigned = shards[args.worker_index]
    rng = np.random.default_rng(args.seed + args.worker_index)
    chosen = np.sort(rng.choice(populations[args.worker_index], target, replace=False))
    model, tokenizer = load_model()
    collector = ActivationSampler(model)
    cursor = 0
    try:
        for progress, row in enumerate(assigned, 1):
            token_ids, start, end = CALIBRATE.model_input(tokenizer, row)
            count = end - start
            if count != int(row["reasoning_token_count"]):
                raise RuntimeError(f"Reasoning-token mismatch for sample {row['sample_id']}")
            local = chosen[(chosen >= cursor) & (chosen < cursor + count)] - cursor
            collector.selected = start + local
            run_chunks(model, token_ids, collector, args.chunk_size)
            cursor += count
            print(
                f"[background {args.worker_index}] {progress}/{len(assigned)} "
                f"sample={row['sample_id']} selected={len(local)}",
                flush=True,
            )
    finally:
        collector.close()
    if cursor != populations[args.worker_index]:
        raise RuntimeError("Background token population changed")
    for layer in LAYERS:
        values = np.concatenate(collector.values[layer])
        if values.shape != (target, 4096):
            raise RuntimeError(f"Unexpected L{layer} background shape {values.shape}")
        np.save(args.output_dir / f"background.worker-{args.worker_index:02d}-L{layer}.npy", values)


def fit_pca(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    metadata = {
        "model": MODEL_ID,
        "layers": list(LAYERS),
        "responses": 789,
        "sampled_reasoning_tokens": args.pca_samples,
        "sample_manifests": [str(path) for path in args.sample_manifests],
        "seed": args.seed,
        "target_explained_variance": args.pca_variance,
        "max_rank": args.pca_rank,
        "components": {},
        "explained_variance": {},
    }
    device = torch.device("cuda:0")
    for layer in LAYERS:
        arrays = [
            np.load(args.output_dir / f"background.worker-{worker:02d}-L{layer}.npy")
            for worker in range(args.num_workers)
        ]
        values = np.concatenate(arrays)
        if values.shape != (args.pca_samples, 4096):
            raise RuntimeError(f"Unexpected merged L{layer} shape {values.shape}")
        matrix = torch.from_numpy(values).to(device=device, dtype=torch.float32)
        q = min(args.pca_rank, matrix.shape[0] - 1, matrix.shape[1])
        _u, singular, vectors = torch.pca_lowrank(matrix, q=q, center=True, niter=4)
        total_variance = matrix.var(dim=0, correction=1).sum()
        cumulative = torch.cumsum(singular.square() / (len(matrix) - 1), dim=0)
        cumulative /= total_variance
        reached = torch.nonzero(cumulative >= args.pca_variance)
        if not len(reached):
            raise RuntimeError(
                f"L{layer}: rank {q} explains only {float(cumulative[-1]):.3f}; "
                "increase --pca-rank"
            )
        components = int(reached[0]) + 1
        np.save(
            args.output_dir / f"basis-L{layer}.npy",
            vectors[:, :components].cpu().numpy().astype(np.float32),
        )
        metadata["components"][str(layer)] = components
        metadata["explained_variance"][str(layer)] = float(cumulative[components - 1])
        print(
            f"L{layer}: {components} PCs explain "
            f"{metadata['explained_variance'][str(layer)]:.1%}",
            flush=True,
        )
        del matrix, _u, singular, vectors
        torch.cuda.empty_cache()
    (args.output_dir / "background_pca.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def raw_diff_vectors() -> torch.Tensor:
    path = hf_hub_download(
        VECTOR_REPO, "diff.safetensors", revision=VECTOR_REVISION
    )
    vectors = load_file(path)["diff"].float()
    if tuple(vectors.shape) != (len(LAYERS), PAIRS, 4096):
        raise RuntimeError(f"Unexpected diff vector shape {tuple(vectors.shape)}")
    return vectors


def orthogonalize_vectors(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projected = vectors - (vectors @ basis) @ basis.T
    norms = projected.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms < 1e-8).any():
        raise RuntimeError("PC removal collapsed or invalidated a concept vector")
    return projected / norms


def load_probe_vectors(background_dir: Path, device: torch.device) -> dict[int, torch.Tensor]:
    raw = raw_diff_vectors()
    probes = {}
    for index, layer in enumerate(LAYERS):
        basis = torch.from_numpy(np.load(background_dir / f"basis-L{layer}.npy")).float()
        if (
            basis.ndim != 2
            or basis.shape[0] != 4096
            or basis.shape[1] < 1
            or not torch.isfinite(basis).all()
        ):
            raise RuntimeError(f"Invalid L{layer} background basis shape or values")
        identity = torch.eye(basis.shape[1])
        if not torch.allclose(basis.T @ basis, identity, atol=2e-4):
            raise RuntimeError(f"L{layer} background basis is not orthonormal")
        probes[layer] = orthogonalize_vectors(raw[index], basis).to(
            device=device, dtype=torch.float32
        )
    return probes


def read_aime_records(results: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (results / "aime_2024.jsonl").read_text().splitlines()
        if line
    ]
    if len(rows) != 30 or len({str(row["id"]) for row in rows}) != 30:
        raise RuntimeError(f"Expected 30 unique saved AIME outputs, found {len(rows)}")
    return sorted(rows, key=lambda row: int(row["id"]))


def exact_saved_sequence(
    tokenizer: Any, record: dict[str, Any], examples: dict[str, dict[str, Any]]
) -> tuple[list[int], list[int], int, int]:
    example = examples[str(record["id"])]
    prompt = EVALUATE.instruction("aime_2024", example["prompt"])
    fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
    if record.get("prompt_sha256") != fingerprint:
        raise RuntimeError(f"Prompt hash mismatch for AIME {record['id']}")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prefix_ids = tokenizer([rendered], return_tensors="pt").input_ids[0].tolist()
    output_ids = tokenizer.encode(record["output"], add_special_tokens=False)
    start, end = thinking_span(tokenizer, output_ids)
    expected_reasoning = int(record["reasoning_token_count"])
    if end > len(output_ids) or abs((end - start) - expected_reasoning) > 2:
        raise RuntimeError(f"Reasoning-token mismatch for AIME {record['id']}")
    return prefix_ids + output_ids, output_ids, start, end


class ProjectionCollector:
    def __init__(
        self,
        model: Any,
        probes: dict[int, torch.Tensor],
        output_dir: Path,
        output_start: int,
        output_tokens: int,
    ) -> None:
        self.model = model
        self.probes = probes
        self.output_start = output_start
        self.output_end = output_start + output_tokens
        self.offset = 0
        self.written = {layer: 0 for layer in LAYERS}
        self.paths = {layer: output_dir / f"L{layer}.tmp.npy" for layer in LAYERS}
        self.arrays = {
            layer: np.lib.format.open_memmap(
                self.paths[layer], mode="w+", dtype=np.float16, shape=(output_tokens, PAIRS)
            )
            for layer in LAYERS
        }
        self.handles = [
            model.model.layers[layer - 1].register_forward_hook(self._hook(layer))
            for layer in LAYERS
        ]

    def _hook(self, layer: int):
        def score(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            lo = max(self.output_start - self.offset, 0)
            hi = min(self.output_end - self.offset, hidden.shape[1])
            if hi <= lo:
                return
            values = hidden[0, lo:hi].float() @ self.probes[layer].T
            destination = self.offset + lo - self.output_start
            cpu = values.to(torch.float16).cpu().numpy()
            self.arrays[layer][destination : destination + len(cpu)] = cpu
            self.written[layer] += len(cpu)

        return score

    def finish(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        expected = self.output_end - self.output_start
        if any(count != expected for count in self.written.values()):
            raise RuntimeError(f"Incomplete projection traces: {self.written}")
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        for layer in LAYERS:
            self.paths[layer].replace(self.paths[layer].with_name(f"L{layer}.npy"))


def summarize_trace(
    trace: Path, record: dict[str, Any], thinking_start: int, thinking_end: int
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows, scales = [], {}
    for layer in LAYERS:
        values = np.load(trace / f"L{layer}.npy", mmap_mode="r")
        reasoning = values[thinking_start:thinking_end].astype(np.float32)
        means = reasoning.mean(axis=0)
        minima = reasoning.min(axis=0)
        maxima = reasoning.max(axis=0)
        scale = float(np.quantile(np.abs(values), 0.99))
        scales[str(layer)] = max(scale, 1e-8)
        rows.extend(
            {
                "benchmark": "aime_2024",
                "id": str(record["id"]),
                "method": "diff",
                "layer": layer,
                "pair": pair,
                "mean_projection": float(means[pair]),
                "min_projection": float(minima[pair]),
                "max_projection": float(maxima[pair]),
                "reasoning_tokens": thinking_end - thinking_start,
                "correct": bool(record["correct"]),
            }
            for pair in range(PAIRS)
        )
    return rows, scales


def trace_complete(trace: Path, output_tokens: int) -> bool:
    if not (trace / "meta.json").exists():
        return False
    try:
        return all(
            np.load(trace / f"L{layer}.npy", mmap_mode="r").shape
            == (output_tokens, PAIRS)
            for layer in LAYERS
        )
    except (FileNotFoundError, ValueError):
        return False


def score_worker(args: argparse.Namespace) -> None:
    records = read_aime_records(args.aime_results)
    assigned = records[args.worker_index :: args.num_workers]
    examples = {row["id"]: row for row in EVALUATE.load_benchmark("aime_2024")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prepared = [
        (record, *exact_saved_sequence(tokenizer, record, examples))
        for record in assigned
    ]
    missing = [
        item
        for item in prepared
        if not trace_complete(
            args.output_dir / "traces" / "aime_2024" / str(item[0]["id"]),
            len(item[2]),
        )
    ]
    model = probes = None
    if missing:
        model, tokenizer = load_model(tokenizer)
        probes = load_probe_vectors(args.background_dir, model.device)
    rows = []
    for progress, (record, token_ids, output_ids, start, end) in enumerate(prepared, 1):
        if model is not None and len(token_ids) > model.config.max_position_embeddings:
            raise RuntimeError(f"AIME {record['id']} exceeds the model context window")
        trace = args.output_dir / "traces" / "aime_2024" / str(record["id"])
        trace.mkdir(parents=True, exist_ok=True)
        reused = trace_complete(trace, len(output_ids))
        if not reused:
            collector = ProjectionCollector(
                model, probes, trace, len(token_ids) - len(output_ids), len(output_ids)
            )
            try:
                run_chunks(model, token_ids, collector, args.chunk_size)
                collector.finish()
            finally:
                for handle in collector.handles:
                    handle.remove()
        summary, scales = summarize_trace(trace, record, start, end)
        rows.extend(summary)
        meta = {
            "id": str(record["id"]),
            "correct": bool(record["correct"]),
            "tokens": [
                tokenizer.decode([token], skip_special_tokens=False) for token in output_ids
            ],
            "token_ids": output_ids,
            "thinking_start": start,
            "thinking_end": end,
            "reasoning_tokens": end - start,
            "source_reasoning_tokens": int(record["reasoning_token_count"]),
            "color_scales": scales,
            "dtype": "float16",
            "method": "diff",
        }
        (trace / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        print(
            f"[AIME {args.worker_index}] {progress}/{len(assigned)} "
            f"id={record['id']} output={len(output_ids)} reasoning={end-start} "
            f"reused={reused}",
            flush=True,
        )
    pd.DataFrame(rows).to_parquet(
        args.output_dir / f"concept_scores.worker-{args.worker_index:02d}.parquet",
        index=False,
    )


def merge_scores(args: argparse.Namespace) -> None:
    scores = pd.concat(
        [
            pd.read_parquet(
                args.output_dir / f"concept_scores.worker-{worker:02d}.parquet"
            )
            for worker in range(args.num_workers)
        ],
        ignore_index=True,
    )
    expected = 30 * len(LAYERS) * PAIRS
    if len(scores) != expected or scores[["id", "layer", "pair"]].duplicated().any():
        raise RuntimeError(f"Expected {expected} unique score rows, found {len(scores)}")
    if not np.isfinite(
        scores[["mean_projection", "min_projection", "max_projection"]].to_numpy()
    ).all():
        raise RuntimeError("Projection scores contain non-finite values")
    scores.sort_values(["id", "layer", "pair"]).to_parquet(
        args.output_dir / "concept_scores-aime_2024.parquet", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "source_results": str(args.aime_results),
                "background": str(args.background_dir),
                "method": "diff",
                "layers": list(LAYERS),
                "responses": 30,
                "score": "raw residual dot unit diff vector after background-PC removal",
                "aggregate_scope": "thinking tokens",
                "trace_scope": "full saved assistant output",
                "trace_dtype": "float16",
            },
            indent=2,
        )
        + "\n"
    )


def launch(args: argparse.Namespace, phase: str) -> None:
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(
            f"Requested {args.num_workers} workers, visible GPUs: {gpu_ids}"
        )
    processes = []
    for worker in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker]
        command = [sys.executable, "-u", str(Path(__file__).resolve()), phase]
        if phase == "fit-background":
            command += [
                "--sample-manifests",
                *map(str, args.sample_manifests),
                "--pca-samples",
                str(args.pca_samples),
                "--pca-rank",
                str(args.pca_rank),
                "--pca-variance",
                str(args.pca_variance),
                "--seed",
                str(args.seed),
            ]
        else:
            command += [
                "--aime-results",
                str(args.aime_results),
                "--background-dir",
                str(args.background_dir),
            ]
        command += [
            "--output-dir",
            str(args.output_dir),
            "--num-workers",
            str(args.num_workers),
            "--chunk-size",
            str(args.chunk_size),
            "--worker-index",
            str(worker),
        ]
        print(f"Launching {phase} worker {worker} on GPU {gpu_ids[worker]}", flush=True)
        processes.append(subprocess.Popen(command, env=environment))
    failures = []
    for worker, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append((worker, code))
    if failures:
        raise RuntimeError(f"{phase} workers failed: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    background = subparsers.add_parser("fit-background")
    background.add_argument("--sample-manifests", type=Path, nargs="+", required=True)
    background.add_argument("--pca-samples", type=int, default=20_000)
    background.add_argument("--pca-rank", type=int, default=512)
    background.add_argument("--pca-variance", type=float, default=0.5)
    background.add_argument("--seed", type=int, default=2026)
    background.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Fit PCA from existing worker arrays without replaying responses.",
    )
    score = subparsers.add_parser("score-aime")
    score.add_argument("--aime-results", type=Path, required=True)
    score.add_argument("--background-dir", type=Path, required=True)
    score.add_argument(
        "--reuse-workers",
        action="store_true",
        help="Merge existing worker score files without replaying responses.",
    )
    for command in (background, score):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--chunk-size", type=int, default=256)
        command.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_workers < 1 or args.chunk_size < 1:
        parser.error("--num-workers and --chunk-size must be positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    if args.phase == "fit-background":
        if args.pca_samples < 2 or args.pca_rank < 1:
            parser.error("--pca-samples must be at least 2 and --pca-rank positive")
        if not 0 < args.pca_variance < 1:
            parser.error("--pca-variance must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        if args.phase == "fit-background":
            background_worker(args)
        else:
            score_worker(args)
        return
    if args.phase == "fit-background":
        if not args.reuse_samples:
            launch(args, args.phase)
        fit_pca(args)
    else:
        if not args.reuse_workers:
            launch(args, args.phase)
        merge_scores(args)
    print(f"Completed {args.phase}: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
