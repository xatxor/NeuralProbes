"""Run a local Qwen3 GCG safety experiment in resumable stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "01_eval" if (PROJECT / "01_eval").exists() else ROOT))
from concept_analysis import AnalysisWriter, ConceptScorer, LAYERS, VECTOR_REVISION  # noqa: E402
from gcg import GCGConfig, optimize  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
DATASET_ID = "walledai/AdvBench"
CONDITIONS = ("baseline", "gcg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "attack", "generate", "judge", "report", "all"))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "advbench-faster-gcg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=20)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--suffix-tokens", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=32)
    parser.add_argument("--distance-penalty", type=float, default=10.0)
    parser.add_argument("--candidate-temperature", type=float, default=0.1)
    parser.add_argument("--activation-chunk-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        parser.error("worker-index must be in [0, num-workers)")
    return args


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def stratified_split(rows: list[dict[str, str]], train_count: int, test_count: int, seed: int) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    rng = np.random.default_rng(seed)
    for group in groups.values():
        rng.shuffle(group)
    categories = sorted(groups)
    picked: list[dict[str, str]] = []
    cursors = defaultdict(int)
    while len(picked) < train_count + test_count:
        progressed = False
        for category in categories:
            if len(picked) >= train_count + test_count:
                break
            cursor = cursors[category]
            if cursor < len(groups[category]):
                picked.append(groups[category][cursor])
                cursors[category] += 1
                progressed = True
        if not progressed:
            raise ValueError("Dataset is too small for requested split")
    return {"train": picked[:train_count], "test": picked[train_count:]}


def prepare(args: argparse.Namespace) -> dict[str, list[dict[str, str]]]:
    split_path = args.output / "split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())
    dataset = load_dataset(DATASET_ID, split="train")
    rows = [{"id": str(index), "prompt": row["prompt"], "target": row["target"], "category": "AdvBench"} for index, row in enumerate(dataset)]
    split = stratified_split(rows, args.train_samples, args.test_samples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(split_path, split)
    write_json(args.output / "metadata.json", {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "seed": args.seed,
        "thinking_enabled": False,
        "concept_method": "diff",
        "layers": list(LAYERS),
        "vector_revision": VECTOR_REVISION,
    })
    return split


def initialize_distributed() -> None:
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_model() -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 GCG requires a CUDA GPU")
    initialize_distributed()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    return model, tokenizer, device


def run_attack(args: argparse.Namespace, split: dict[str, list[dict[str, str]]]) -> None:
    model, tokenizer, device = load_model()
    optimize(
        model,
        tokenizer,
        split["train"],
        args.output,
        GCGConfig(
            args.steps, args.suffix_tokens, args.batch_size, args.topk,
            args.candidate_chunk_size, args.seed, args.distance_penalty,
            args.candidate_temperature,
        ),
        device,
    )


def generate_one(model: Any, tokenizer: Any, prompt: str, scorer: ConceptScorer, condition: str, sample_id: str, max_new_tokens: int | None) -> tuple[str, Any, float, str]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    scorer.begin(condition, sample_id)
    started = time.perf_counter()
    try:
        generation_args = {"do_sample": False}
        generation_args["max_new_tokens" if max_new_tokens is not None else "max_length"] = max_new_tokens or model.config.max_position_embeddings
        output = model.generate(**inputs, **generation_args)
    except Exception:
        scorer.cancel()
        raise
    continuation = output[0][inputs.input_ids.shape[1] :]
    analysis = scorer.finish(continuation, condition, sample_id, span_mode="continuation")
    eos_ids = {tokenizer.eos_token_id} if isinstance(tokenizer.eos_token_id, int) else set(tokenizer.eos_token_id or [])
    reason = "eos" if continuation.numel() and continuation[-1].item() in eos_ids else "limit"
    return tokenizer.decode(continuation, skip_special_tokens=False), analysis, time.perf_counter() - started - analysis.analysis_seconds, reason


def run_generate(args: argparse.Namespace, split: dict[str, list[dict[str, str]]]) -> None:
    attack = json.loads((args.output / "attack.json").read_text())
    suffixes = {"baseline": "", "gcg": attack["suffix"]}
    model, tokenizer, _device = load_model()
    rows = split["test"][args.worker_index :: args.num_workers]
    completed = {(row["condition"], row["id"]) for row in jsonl(args.output / "responses.jsonl")}
    scorer = ConceptScorer(model, tokenizer, model.device, args.output / "traces", pair_ids=list(range(1036)), activation_chunk_size=args.activation_chunk_size, methods=("diff",))
    writers = {condition: AnalysisWriter(args.output, f"-{condition}.worker-{args.worker_index:02d}") for condition in CONDITIONS}
    with (args.output / "responses.jsonl").open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            for condition in CONDITIONS:
                if (condition, row["id"]) in completed:
                    continue
                output, analysis, seconds, finish_reason = generate_one(
                    model, tokenizer, row["prompt"] + suffixes[condition], scorer, condition, row["id"], args.max_new_tokens
                )
                writers[condition].add(analysis)
                record = row | {
                    "condition": condition,
                    "output": output,
                    "response_tokens": analysis.reasoning_token_count,
                    "response_status": analysis.reasoning_status,
                    "generation_seconds": seconds,
                    "analysis_seconds": analysis.analysis_seconds,
                    "finish_reason": finish_reason,
                    "suffix": suffixes[condition],
                    "prompt_sha256": hashlib.sha256((row["prompt"] + suffixes[condition]).encode()).hexdigest(),
                    "worker": args.worker_index,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            print(f"worker {args.worker_index}: {index}/{len(rows)}", flush=True)
    for writer in writers.values():
        writer.close()


def run_judge(args: argparse.Namespace) -> None:
    rows = list({(row["condition"], row["id"]): row for row in jsonl(args.output / "responses.jsonl")}.values())
    existing = {(row["condition"], row["id"]) for row in jsonl(args.output / "judgments.jsonl")}
    todo = [row for row in rows if (row["condition"], row["id"]) not in existing]
    if not todo:
        return
    try:
        from strong_reject.evaluate import evaluate_dataset
    except ImportError as error:
        raise SystemExit("Install StrongREJECT first: pip install git+https://github.com/dsbowen/strong_reject.git@main") from error
    dataset = Dataset.from_list([{"forbidden_prompt": row["prompt"], "response": row["output"]} for row in todo])
    judged = evaluate_dataset(dataset, ["strongreject_finetuned"], batch_size=8)
    with (args.output / "judgments.jsonl").open("a", encoding="utf-8") as stream:
        for row, judged_row in zip(todo, judged, strict=True):
            score = next(value for key, value in judged_row.items() if key.endswith("score"))
            stream.write(json.dumps({"id": row["id"], "condition": row["condition"], "strongreject_score": float(score)}, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    initialize_distributed()
    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        split = json.loads((args.output / "split.json").read_text())
    else:
        split = prepare(args)
        if dist.is_initialized():
            dist.barrier()
    if args.stage == "prepare":
        return
    try:
        if args.stage in ("attack", "all"):
            run_attack(args, split)
        if args.stage in ("generate", "all"):
            run_generate(args, split)
        if args.stage in ("judge", "all"):
            run_judge(args)
        if args.stage in ("report", "all"):
            from report import build_report
            build_report(args.output)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
