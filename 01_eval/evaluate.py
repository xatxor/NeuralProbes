"""Evaluate Qwen/Qwen3-8B with thinking enabled on three reasoning benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from concept_analysis import AnalysisWriter, ConceptScorer, VERSION as ANALYSIS_VERSION

MODEL_ID = "Qwen/Qwen3-8B"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("aime_2024", "math_500", "gpqa_diamond", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of data-parallel GPU workers. Each worker loads one model replica.",
    )
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--concept-analysis", action="store_true", help="Score reasoning tokens against Qwen3 concept vectors.")
    parser.add_argument("--highlights-per-sign", type=int, default=3)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    if args.highlights_per_sign < 1:
        parser.error("--highlights-per-sign must be at least 1")
    return args


def final_answer(text: str) -> str:
    """Prefer the final boxed answer, otherwise the final non-empty line."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().rstrip(".").lower()


def extract_choice(text: str) -> str | None:
    # Prefer an explicit final answer, then take the last standalone A-D marker.
    candidates = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**([A-D])\b", text, re.I)
    if not candidates:
        candidates = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
    return candidates[-1].upper() if candidates else None


def math_equal(prediction: str, reference: str) -> bool | None:
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError:
        return None
    try:
        gold = parse(reference, extraction_config=[LatexExtractionConfig()])
        pred = parse(prediction, extraction_config=[LatexExtractionConfig()])
        return bool(verify(gold, pred))
    except Exception:
        return False


def load_benchmark(name: str) -> list[dict[str, Any]]:
    if name == "aime_2024":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [{"id": str(i), "prompt": row["problem"], "answer": str(row["answer"])} for i, row in enumerate(dataset)]
    if name == "math_500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"id": str(i), "prompt": row["problem"], "answer": row["solution"]} for i, row in enumerate(dataset)]
    if name == "gpqa_diamond":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rows = []
        for i, row in enumerate(dataset):
            options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            random.Random(i).shuffle(options)
            correct = "ABCD"[options.index(row["Correct Answer"])]
            rendered = "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options, strict=True))
            rows.append({"id": str(i), "prompt": f"{row['Question']}\n\n{rendered}", "answer": correct})
        return rows
    raise ValueError(name)


def instruction(name: str, question: str) -> str:
    if name == "math_500":
        # Official prompt_template from HuggingFaceH4/MATH-500/eval.yaml.
        return (
            "Solve the following math problem efficiently and clearly. The last line "
            "of your response should be of the following format: 'Therefore, the final "
            "answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) "
            "where ANSWER is just the final number or expression that solves the "
            f"problem. Think step by step before answering.\n\n{question}"
        )
    suffix = {
        "aime_2024": "Give the final integer answer clearly, preferably as \\boxed{answer}.",
        "gpqa_diamond": "Reason carefully, then end with only the letter of the correct option (A, B, C, or D).",
    }[name]
    return f"Solve the following problem. {suffix}\n\n{question}"


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    scorer: ConceptScorer | None = None,
    benchmark: str = "",
    example_id: str = "",
    highlight_sink: Any | None = None,
    selected_sink: Any | None = None,
) -> tuple[str, Any | None]:
    messages = [{"role": "user", "content": prompt}]
    # Qwen3's native switch; do not omit it, even if the model defaults change.
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    # Let the model generate until it emits EOS. The architecture's context
    # window is the only upper bound, preventing positions the model cannot use.
    if scorer is not None:
        scorer.begin()
    try:
        output = model.generate(**inputs, max_length=model.config.max_position_embeddings, do_sample=False)
    except Exception:
        if scorer is not None:
            for handle in scorer.handles:
                handle.remove()
            scorer.handles = []
        raise
    continuation = output[0][inputs.input_ids.shape[1] :]
    analysis = scorer.finish(continuation, benchmark, example_id, highlight_sink, selected_sink) if scorer is not None else None
    return tokenizer.decode(continuation, skip_special_tokens=False), analysis


def score(name: str, output: str, answer: str) -> bool | None:
    if name == "gpqa_diamond":
        return extract_choice(output) == answer
    if name == "math_500":
        return math_equal(output, answer)
    return normalise(final_answer(output)) == normalise(answer)


def benchmarks_for(args: argparse.Namespace) -> tuple[str, ...]:
    if args.benchmark == "all":
        return ("aime_2024", "math_500", "gpqa_diamond")
    return (args.benchmark,)


def result_path(name: str, args: argparse.Namespace) -> Path:
    if args.worker_index is None:
        return RESULTS / f"{name}.jsonl"
    return RESULTS / f"{name}.worker-{args.worker_index:02d}-of-{args.num_workers:02d}.jsonl"


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prompt_fingerprint(name: str, example: dict[str, Any]) -> str:
    prompt = instruction(name, example["prompt"])
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def matching_records(
    name: str,
    examples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    require_analysis: bool = False,
) -> list[dict[str, Any]]:
    expected_fingerprints = {
        example["id"]: prompt_fingerprint(name, example) for example in examples
    }
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        expected = expected_fingerprints.get(record["id"])
        if (
            expected is not None
            and record.get("prompt_sha256") == expected
            and record.get("model") == MODEL_ID
            and (not require_analysis or record.get("concept_analysis_version") == ANALYSIS_VERSION)
        ):
            records_by_id[record["id"]] = record
    return [records_by_id[example["id"]] for example in examples if example["id"] in records_by_id]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize(name: str, records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    scored = [row["correct"] for row in records if row["correct"] is not None]
    return {
        "benchmark": name,
        "examples": len(records),
        "scored": len(scored),
        "accuracy": sum(scored) / len(scored) if scored else None,
        "generation_seconds": sum(row.get("generation_seconds", 0.0) for row in records),
        "run_wall_seconds": wall_seconds,
    }


def run(name: str, model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    path = result_path(name, args)
    examples = load_benchmark(name)
    if args.limit is not None:
        examples = examples[: args.limit]
    if args.worker_index is not None:
        examples = examples[args.worker_index :: args.num_workers]
    completed = {
        row["id"]
        for row in matching_records(name, examples, read_records(path), require_analysis=args.concept_analysis)
    }
    worker_label = f"worker {args.worker_index}" if args.worker_index is not None else "worker 0"
    scorer = (
        ConceptScorer(model, tokenizer, model.device, args.highlights_per_sign)
        if args.concept_analysis
        else None
    )
    writer = AnalysisWriter(RESULTS, f"-{path.stem}") if args.concept_analysis else None
    with path.open("a", encoding="utf-8") as handle:
        for n, example in enumerate(examples, 1):
            if example["id"] in completed:
                continue
            formatted_prompt = instruction(name, example["prompt"])
            example_started = time.perf_counter()
            output, analysis = generate(
                model, tokenizer, formatted_prompt, scorer, name, example["id"], writer.add_highlights if writer else None, writer.add_selected if writer else None
            )
            generation_seconds = time.perf_counter() - example_started
            record = {
                "id": example["id"],
                "model": MODEL_ID,
                "thinking_enabled": True,
                "prompt_sha256": hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest(),
                "output": output,
                "reference": example["answer"],
                "correct": score(name, output, example["answer"]),
                "generation_seconds": generation_seconds,
            }
            if analysis is not None:
                writer.add(analysis)
                record.update(
                    {
                        "concept_analysis_version": ANALYSIS_VERSION,
                        "reasoning_token_count": analysis.reasoning_token_count,
                        "reasoning_status": analysis.reasoning_status,
                        "reasoning_tokens": analysis.tokens,
                        "activation_color_scales": analysis.color_scales,
                    }
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{worker_label}] {name}: {n}/{len(examples)} | id={example['id']} "
                f"| correct={record['correct']} "
                f"| generation_seconds={generation_seconds:.1f}",
                flush=True,
            )
    if writer is not None:
        writer.close()
    records = matching_records(name, examples, read_records(path))
    write_jsonl_atomic(path, records)
    summary = summarize(name, records, time.perf_counter() - run_started)
    if args.worker_index is None:
        (RESULTS / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{worker_label}] {json.dumps(summary)}", flush=True)


def visible_gpu_ids() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [device.strip() for device in configured.split(",") if device.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def worker_command(args: argparse.Namespace, worker_index: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--benchmark",
        args.benchmark,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--worker-index",
        str(worker_index),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.concept_analysis:
        command.extend(["--concept-analysis", "--highlights-per-sign", str(args.highlights_per_sign)])
    return command


def seed_missing_shards(args: argparse.Namespace) -> None:
    """Seed new shard files from a resumable single-worker result, if present."""
    RESULTS.mkdir(exist_ok=True)
    for name in benchmarks_for(args):
        canonical_records = read_records(RESULTS / f"{name}.jsonl")
        if not canonical_records:
            continue
        examples = load_benchmark(name)
        if args.limit is not None:
            examples = examples[: args.limit]
        worker_for_id = {
            example["id"]: position % args.num_workers for position, example in enumerate(examples)
        }
        compatible_records = matching_records(name, examples, canonical_records, require_analysis=args.concept_analysis)
        for worker_index in range(args.num_workers):
            shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker_index})
            shard = result_path(name, shard_args)
            if shard.exists():
                continue
            seeded = [
                record
                for record in compatible_records
                if worker_for_id.get(record["id"]) == worker_index
            ]
            write_jsonl_atomic(shard, seeded)


def merge_shards(name: str, args: argparse.Namespace, wall_seconds: float) -> dict[str, Any]:
    examples = load_benchmark(name)
    if args.limit is not None:
        examples = examples[: args.limit]
    expected_ids = [example["id"] for example in examples]
    records_by_id: dict[str, dict[str, Any]] = {}
    for worker_index in range(args.num_workers):
        worker_examples = examples[worker_index :: args.num_workers]
        shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker_index})
        shard_records = read_records(result_path(name, shard_args))
        for record in matching_records(name, worker_examples, shard_records, require_analysis=args.concept_analysis):
            records_by_id[record["id"]] = record
    missing = [example_id for example_id in expected_ids if example_id not in records_by_id]
    if missing:
        raise RuntimeError(f"{name}: workers finished but {len(missing)} examples are missing: {missing[:10]}")
    records = [records_by_id[example_id] for example_id in expected_ids]
    write_jsonl_atomic(RESULTS / f"{name}.jsonl", records)
    if args.concept_analysis:
        merge_analysis_files(name, args)
    summary = summarize(name, records, wall_seconds)
    (RESULTS / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def merge_analysis_files(name: str, args: argparse.Namespace) -> None:
    """Concatenate worker parquet row groups without loading the analysis into RAM."""
    import pyarrow.parquet as pq

    for stem in ("concept_scores", "token_highlights", "selected_token_activations"):
        sources = [
            RESULTS / f"{stem}-{name}.worker-{worker:02d}-of-{args.num_workers:02d}.parquet"
            for worker in range(args.num_workers)
        ]
        if not all(path.exists() for path in sources):
            missing = [str(path) for path in sources if not path.exists()]
            raise RuntimeError(f"Missing concept-analysis parquet files: {missing}")
        destination = RESULTS / f"{stem}-{name}.parquet"
        writer = None
        try:
            for source in sources:
                reader = pq.ParquetFile(source)
                if writer is None:
                    writer = pq.ParquetWriter(destination, reader.schema_arrow, compression="zstd")
                for batch in reader.iter_batches():
                    writer.write_batch(batch)
        finally:
            if writer is not None:
                writer.close()


def launch_data_parallel(args: argparse.Namespace) -> None:
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(
            f"Requested {args.num_workers} workers, but only {len(gpu_ids)} GPUs are visible: {gpu_ids}"
        )
    seed_missing_shards(args)
    started = time.perf_counter()
    processes: list[subprocess.Popen[bytes]] = []
    for worker_index in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_index]
        command = worker_command(args, worker_index)
        print(
            f"Launching worker {worker_index} on physical GPU {gpu_ids[worker_index]}: "
            f"{' '.join(command)}",
            flush=True,
        )
        processes.append(subprocess.Popen(command, env=environment))
    return_codes = [process.wait() for process in processes]
    failures = [
        (worker_index, return_code)
        for worker_index, return_code in enumerate(return_codes)
        if return_code != 0
    ]
    if failures:
        raise RuntimeError(f"Data-parallel workers failed: {failures}")
    wall_seconds = time.perf_counter() - started
    for name in benchmarks_for(args):
        merge_shards(name, args, wall_seconds)


def main() -> None:
    args = parse_args()
    if args.worker_index is None and args.num_workers > 1:
        launch_data_parallel(args)
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    for benchmark in benchmarks_for(args):
        run(benchmark, model, tokenizer, args)


if __name__ == "__main__":
    main()
