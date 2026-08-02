"""Fit neutral PCs on Alpaca responses and score saved AdvBench responses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LAYERS = (11, 14, 18, 22, 25)
MODEL_ID = "Qwen/Qwen3-8B"
PAIRS = 1036


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


PROJECTION = module("neutral_projection", PROJECT / "04_projection_probes" / "pipeline.py")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def load_model(tokenizer: Any | None = None):
    return PROJECTION.load_model(tokenizer)


def render(tokenizer: Any, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def alpaca_rows(seed: int, samples: int) -> list[dict[str, str]]:
    data = load_dataset("tatsu-lab/alpaca", split="train")
    indices = np.random.default_rng(seed).choice(len(data), samples, replace=False)
    return [
        {
            "id": str(int(index)),
            "prompt": str(data[int(index)]["instruction"])
            + (f"\n\nInput:\n{data[int(index)]['input']}" if data[int(index)]["input"] else ""),
        }
        for index in indices
    ]


def generate_worker(args: argparse.Namespace) -> None:
    rows = alpaca_rows(args.seed, args.samples)[args.worker :: args.workers]
    path = args.output / f"responses.worker-{args.worker:02d}.jsonl"
    done = {row["id"] for row in jsonl(path)}
    model, tokenizer = load_model()
    with path.open("a") as stream:
        for progress, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            prefix = torch.tensor([render(tokenizer, row["prompt"])], device=model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(
                    input_ids=prefix,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            output_ids = result[0, prefix.shape[1] :].tolist()
            stream.write(json.dumps(row | {
                "output": tokenizer.decode(output_ids, skip_special_tokens=False),
                "output_ids": output_ids,
                "response_tokens": len(output_ids),
                "generation_seconds": time.perf_counter() - started,
            }, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[neutral {args.worker}] {progress}/{len(rows)} id={row['id']} tokens={len(output_ids)}", flush=True)


def merge_neutral(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for worker in range(args.workers) for row in jsonl(args.output / f"responses.worker-{worker:02d}.jsonl")]
    if len(rows) != args.samples or len({row["id"] for row in rows}) != args.samples:
        raise RuntimeError(f"Expected {args.samples} unique Alpaca responses, found {len(rows)}")
    rows.sort(key=lambda row: int(row["id"]))
    with (args.output / "responses.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def activation_worker(args: argparse.Namespace) -> None:
    rows = jsonl(args.output / "responses.jsonl")[args.worker :: args.workers]
    population = sum(row["response_tokens"] for row in rows)
    counts = PROJECTION.allocate_counts(
        [sum(row["response_tokens"] for row in jsonl(args.output / "responses.jsonl")[worker :: args.workers]) for worker in range(args.workers)],
        min(args.pca_tokens, sum(row["response_tokens"] for row in jsonl(args.output / "responses.jsonl"))),
    )
    target = counts[args.worker]
    chosen = np.sort(np.random.default_rng(args.seed + args.worker).choice(population, target, replace=False))
    model, tokenizer = load_model()
    collector = PROJECTION.ActivationSampler(model)
    cursor = 0
    try:
        for progress, row in enumerate(rows, 1):
            prefix_ids = render(tokenizer, row["prompt"])
            output_ids = list(map(int, row["output_ids"]))
            local = chosen[(chosen >= cursor) & (chosen < cursor + len(output_ids))] - cursor
            collector.selected = len(prefix_ids) + local
            PROJECTION.run_chunks(model, prefix_ids + output_ids, collector, args.chunk_size)
            cursor += len(output_ids)
            print(f"[activations {args.worker}] {progress}/{len(rows)} selected={len(local)}", flush=True)
    finally:
        collector.close()
    for layer in LAYERS:
        values = np.concatenate(collector.values[layer])
        if values.shape != (target, 4096):
            raise RuntimeError(f"Unexpected L{layer} shape {values.shape}")
        np.save(args.output / f"background.worker-{args.worker:02d}-L{layer}.npy", values)


def fit_pca(args: argparse.Namespace) -> None:
    fit_args = argparse.Namespace(
        output_dir=args.output,
        num_workers=args.workers,
        pca_samples=min(args.pca_tokens, sum(row["response_tokens"] for row in jsonl(args.output / "responses.jsonl"))),
        pca_rank=args.pca_rank,
        pca_variance=0.5,
        seed=args.seed,
        sample_manifests=[],
    )
    PROJECTION.fit_pca(fit_args)
    metadata = json.loads((args.output / "background_pca.json").read_text())
    metadata.update({"dataset": "tatsu-lab/alpaca", "responses": args.samples, "scope": "assistant response tokens"})
    (args.output / "background_pca.json").write_text(json.dumps(metadata, indent=2) + "\n")


def score_worker(args: argparse.Namespace) -> None:
    all_rows = jsonl(args.advbench / "responses.jsonl")
    unique = {(row["condition"], row["id"]): row for row in all_rows}
    rows = sorted(unique.values(), key=lambda row: (int(row["id"]), row["condition"]))[args.worker :: args.workers]
    model, tokenizer = load_model()
    probes = PROJECTION.load_probe_vectors(args.neutral, model.device)
    summaries = []
    for progress, row in enumerate(rows, 1):
        prompt = row["prompt"] + row.get("suffix", "")
        prefix_ids, output_ids = render(tokenizer, prompt), tokenizer.encode(row["output"], add_special_tokens=False)
        trace = args.output / "traces" / row["condition"] / row["id"]
        trace.mkdir(parents=True, exist_ok=True)
        collector = PROJECTION.ProjectionCollector(model, probes, trace, len(prefix_ids), len(output_ids))
        try:
            PROJECTION.run_chunks(model, prefix_ids + output_ids, collector, args.chunk_size)
            collector.finish()
        finally:
            for handle in collector.handles:
                handle.remove()
        for layer in LAYERS:
            (trace / f"L{layer}.npy").replace(trace / f"diff-L{layer}.npy")
        for layer in LAYERS:
            values = np.load(trace / f"diff-L{layer}.npy", mmap_mode="r").astype(np.float32)
            means = values.mean(axis=0)
            summaries.extend({
                "id": row["id"], "condition": row["condition"], "layer": layer,
                "pair": pair, "mean_activation": float(means[pair]),
            } for pair in range(PAIRS))
        (trace / "meta.json").write_text(json.dumps({
            "tokens": [tokenizer.decode([token], skip_special_tokens=False) for token in output_ids],
            "token_ids": output_ids, "span_start": 0, "span_end": len(output_ids),
            "method": "diff", "score": "signed residual dot unit PCA-projected concept vector",
        }, ensure_ascii=False))
        print(f"[score {args.worker}] {progress}/{len(rows)} {row['condition']}:{row['id']}", flush=True)
    pd.DataFrame(summaries).to_parquet(args.output / f"scores.worker-{args.worker:02d}.parquet", index=False)


def launch(args: argparse.Namespace, mode: str) -> None:
    processes = []
    for worker in range(args.workers):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker)
        command = [sys.executable, "-u", __file__, mode, "--output", str(args.output), "--workers", str(args.workers), "--worker", str(worker), "--seed", str(args.seed)]
        if mode in {"generate-worker", "activation-worker"}:
            command += ["--samples", str(args.samples), "--max-new-tokens", str(args.max_new_tokens), "--pca-tokens", str(args.pca_tokens), "--chunk-size", str(args.chunk_size)]
        else:
            command += ["--neutral", str(args.neutral), "--advbench", str(args.advbench), "--chunk-size", str(args.chunk_size)]
        processes.append(subprocess.Popen(command, env=env))
    failures = [(index, process.wait()) for index, process in enumerate(processes)]
    failures = [item for item in failures if item[1]]
    if failures:
        raise RuntimeError(f"{mode} failed: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("neutral", "generate-worker", "activation-worker", "score", "score-worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neutral", type=Path)
    parser.add_argument("--advbench", type=Path)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pca-tokens", type=int, default=40_000)
    parser.add_argument("--pca-rank", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "neutral":
        launch(args, "generate-worker")
        merge_neutral(args)
        launch(args, "activation-worker")
        fit_pca(args)
    elif args.mode == "score":
        if not args.neutral or not args.advbench:
            raise SystemExit("--neutral and --advbench are required")
        launch(args, "score-worker")
        frames = [pd.read_parquet(args.output / f"scores.worker-{worker:02d}.parquet") for worker in range(args.workers)]
        pd.concat(frames, ignore_index=True).to_parquet(args.output / "concept_scores.parquet", index=False)
    elif args.mode == "generate-worker":
        generate_worker(args)
    elif args.mode == "activation-worker":
        activation_worker(args)
    else:
        score_worker(args)


if __name__ == "__main__":
    main()
