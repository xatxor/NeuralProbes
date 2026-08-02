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

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from concept_analysis import (
    DEFAULT_LAYERS,
    DEFAULT_METHODS,
    LAYERS as AVAILABLE_LAYERS,
    METHODS as AVAILABLE_METHODS,
    AnalysisWriter,
    ConceptScorer,
    VERSION as ANALYSIS_VERSION,
)

MODEL_ID = "Qwen/Qwen3-8B"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


class UniqueNGramLoopDetector(StoppingCriteria):
    """Stop after sustained low unique-n-gram diversity, preserving a tail for analysis."""

    def __init__(
        self,
        prompt_tokens: int,
        *,
        ngram_size: int,
        window_tokens: int,
        unique_ratio_threshold: float,
        check_every: int,
        consecutive_windows: int,
        min_new_tokens: int,
        extra_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.ngram_size = ngram_size
        self.window_tokens = window_tokens
        self.unique_ratio_threshold = unique_ratio_threshold
        self.check_every = check_every
        self.consecutive_windows = consecutive_windows
        self.min_new_tokens = min_new_tokens
        self.extra_tokens = extra_tokens

        self.last_checked_tokens = 0
        self.low_ratio_checks = 0
        self.last_unique_ratio: float | None = None
        self.detected = False
        self.forced_stop = False
        self.window_start_token: int | None = None
        self.detection_token: int | None = None
        self.detection_unique_ratio: float | None = None
        self.stop_after_token: int | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        del scores, kwargs
        generated_tokens = int(input_ids.shape[1] - self.prompt_tokens)

        if self.detected:
            if self.stop_after_token is not None and generated_tokens >= self.stop_after_token:
                self.forced_stop = True
                return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if generated_tokens < max(self.min_new_tokens, self.window_tokens):
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if generated_tokens - self.last_checked_tokens < self.check_every:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.last_checked_tokens = generated_tokens

        window = input_ids[0, -self.window_tokens :].detach().cpu().tolist()
        total_ngrams = len(window) - self.ngram_size + 1
        if total_ngrams <= 0:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        unique_ngrams = len(
            {
                tuple(window[index : index + self.ngram_size])
                for index in range(total_ngrams)
            }
        )
        ratio = unique_ngrams / total_ngrams
        self.last_unique_ratio = ratio

        if ratio < self.unique_ratio_threshold:
            self.low_ratio_checks += 1
        else:
            self.low_ratio_checks = 0

        if self.low_ratio_checks >= self.consecutive_windows:
            self.detected = True
            self.window_start_token = max(0, generated_tokens - self.window_tokens)
            self.detection_token = generated_tokens
            self.detection_unique_ratio = ratio
            self.stop_after_token = generated_tokens + self.extra_tokens

        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def metadata(self, final_generated_tokens: int) -> dict[str, Any]:
        preserved_tail = (
            max(0, final_generated_tokens - self.detection_token)
            if self.detection_token is not None
            else 0
        )
        return {
            "enabled": True,
            "detected": self.detected,
            "forced_stop": self.forced_stop,
            "ngram_size": self.ngram_size,
            "window_tokens": self.window_tokens,
            "unique_ratio_threshold": self.unique_ratio_threshold,
            "check_every_tokens": self.check_every,
            "consecutive_windows": self.consecutive_windows,
            "min_new_tokens": self.min_new_tokens,
            "configured_extra_tokens": self.extra_tokens,
            "window_start_token": self.window_start_token,
            "detection_token": self.detection_token,
            "stop_token": final_generated_tokens,
            "preserved_tail_tokens": preserved_tail,
            "detection_unique_ngram_ratio": self.detection_unique_ratio,
            "last_unique_ngram_ratio": self.last_unique_ratio,
        }


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
    parser.add_argument("--concept-analysis", action="store_true", help="Score generated response tokens against Qwen3 concept vectors and summarize the reasoning span.")
    parser.add_argument(
        "--activation-chunk-size",
        type=int,
        default=512,
        help="Generated-token activations retained on GPU before cosine scoring (default: 512).",
    )
    parser.add_argument(
        "--concept-methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated concept-vector methods selected from "
            f"{','.join(AVAILABLE_METHODS)} (default: {','.join(DEFAULT_METHODS)})."
        ),
    )
    parser.add_argument(
        "--concept-layers",
        default=",".join(map(str, DEFAULT_LAYERS)),
        help=(
            "Comma-separated residual-stream layers selected from "
            f"{','.join(map(str, AVAILABLE_LAYERS))} "
            f"(default: {','.join(map(str, DEFAULT_LAYERS))})."
        ),
    )
    parser.add_argument("--concept-pairs", default=None, help="Comma-separated concept pair IDs; default scores all 1036 pairs.")
    parser.add_argument(
        "--disable-loop-detection",
        action="store_true",
        help="Disable online low-diversity loop detection.",
    )
    parser.add_argument("--loop-ngram-size", type=int, default=4)
    parser.add_argument("--loop-window-tokens", type=int, default=1024)
    parser.add_argument("--loop-unique-ratio-threshold", type=float, default=0.20)
    parser.add_argument("--loop-check-every", type=int, default=64)
    parser.add_argument("--loop-consecutive-windows", type=int, default=3)
    parser.add_argument("--loop-min-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--loop-extra-tokens",
        type=int,
        default=512,
        help="Tokens retained after loop detection so several repeated phrases remain analyzable.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    if args.activation_chunk_size < 1:
        parser.error("--activation-chunk-size must be at least 1")
    if args.loop_ngram_size < 1:
        parser.error("--loop-ngram-size must be at least 1")
    if args.loop_window_tokens < args.loop_ngram_size:
        parser.error("--loop-window-tokens must be at least --loop-ngram-size")
    if not 0.0 <= args.loop_unique_ratio_threshold <= 1.0:
        parser.error("--loop-unique-ratio-threshold must be between 0 and 1")
    if args.loop_check_every < 1:
        parser.error("--loop-check-every must be at least 1")
    if args.loop_consecutive_windows < 1:
        parser.error("--loop-consecutive-windows must be at least 1")
    if args.loop_min_new_tokens < 0:
        parser.error("--loop-min-new-tokens must be non-negative")
    if args.loop_extra_tokens < 0:
        parser.error("--loop-extra-tokens must be non-negative")
    args.concept_methods = tuple(
        dict.fromkeys(value.strip() for value in args.concept_methods.split(",") if value.strip())
    )
    if not args.concept_methods or any(method not in AVAILABLE_METHODS for method in args.concept_methods):
        parser.error(
            "--concept-methods must be a non-empty comma-separated subset of "
            + ",".join(AVAILABLE_METHODS)
        )
    try:
        args.concept_layers = tuple(
            sorted({int(value) for value in args.concept_layers.split(",") if value.strip()})
        )
    except ValueError as error:
        parser.error(f"--concept-layers must contain integer layer IDs: {error}")
    if not args.concept_layers or any(layer not in AVAILABLE_LAYERS for layer in args.concept_layers):
        parser.error(
            "--concept-layers must be a non-empty comma-separated subset of "
            + ",".join(map(str, AVAILABLE_LAYERS))
        )
    if args.concept_pairs is not None:
        try:
            args.concept_pair_ids = sorted({int(value) for value in args.concept_pairs.split(",") if value.strip()})
        except ValueError as error:
            parser.error(f"--concept-pairs must contain integer IDs: {error}")
        if not args.concept_pair_ids or min(args.concept_pair_ids) < 0 or max(args.concept_pair_ids) >= 1036:
            parser.error("--concept-pairs IDs must be between 0 and 1035")
    else:
        args.concept_pair_ids = list(range(1036))
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
    loop_options: dict[str, Any] | None = None,
) -> tuple[str, Any | None, dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    # Qwen3's native switch; do not omit it, even if the model defaults change.
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    # Let the model generate until EOS or until a sustained low-diversity loop
    # has been observed and an additional tail has been preserved for analysis.
    loop_detector = (
        UniqueNGramLoopDetector(inputs.input_ids.shape[1], **loop_options)
        if loop_options is not None
        else None
    )
    generation_kwargs: dict[str, Any] = {
        "max_length": model.config.max_position_embeddings,
        "do_sample": False,
    }
    if loop_detector is not None:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([loop_detector])

    if scorer is not None:
        scorer.begin(benchmark, example_id)
    try:
        output = model.generate(**inputs, **generation_kwargs)
    except Exception:
        if scorer is not None:
            scorer.cancel()
        raise
    continuation = output[0][inputs.input_ids.shape[1] :]
    try:
        analysis = scorer.finish(continuation, benchmark, example_id) if scorer is not None else None
    except Exception:
        if scorer is not None:
            scorer.cancel()
        raise
    loop_metadata = (
        loop_detector.metadata(len(continuation))
        if loop_detector is not None
        else {"enabled": False, "detected": False, "forced_stop": False}
    )
    return (
        tokenizer.decode(continuation, skip_special_tokens=False),
        analysis,
        loop_metadata,
    )


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


def trace_complete(
    benchmark: str,
    example_id: str,
    pair_ids: list[int],
    methods: tuple[str, ...],
    layers: tuple[int, ...],
) -> bool:
    root = RESULTS / "traces" / benchmark / str(example_id)
    meta = root / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return False
    return (
        data.get("pair_ids") == pair_ids
        and data.get("dtype") == "float16"
        and set(methods).issubset(set(data.get("methods", AVAILABLE_METHODS)))
        and set(layers).issubset(set(data.get("layers", AVAILABLE_LAYERS)))
        and isinstance(data.get("full_tokens"), list)
        and isinstance(data.get("reasoning_start"), int)
        and isinstance(data.get("reasoning_end"), int)
        and all(
            (root / f"full-{method}-L{layer}.npy").exists()
            for method in methods
            for layer in layers
        )
    )


def matching_records(
    name: str,
    examples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    require_analysis: bool = False,
    concept_pair_ids: list[int] | None = None,
    concept_methods: tuple[str, ...] = DEFAULT_METHODS,
    concept_layers: tuple[int, ...] = DEFAULT_LAYERS,
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
            and (
                not require_analysis
                or record.get("concept_analysis_version") in {7, ANALYSIS_VERSION}
            )
            and (not require_analysis or record.get("concept_pair_ids") == concept_pair_ids)
            and (
                not require_analysis
                or set(concept_methods).issubset(
                    set(record.get("concept_methods", AVAILABLE_METHODS))
                )
            )
            and (
                not require_analysis
                or set(concept_layers).issubset(
                    set(record.get("concept_layers", AVAILABLE_LAYERS))
                )
            )
            and (not require_analysis or record.get("activation_chunk_size") is not None)
            and (
                not require_analysis
                or trace_complete(
                    name,
                    record["id"],
                    concept_pair_ids or [],
                    concept_methods,
                    concept_layers,
                )
            )
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
        "analysis_seconds": sum(row.get("analysis_seconds", 0.0) for row in records),
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
        for row in matching_records(name, examples, read_records(path), require_analysis=args.concept_analysis, concept_pair_ids=args.concept_pair_ids, concept_methods=args.concept_methods, concept_layers=args.concept_layers)
    }
    worker_label = f"worker {args.worker_index}" if args.worker_index is not None else "worker 0"
    scorer = (
        ConceptScorer(
            model,
            tokenizer,
            model.device,
            RESULTS / "traces",
            args.concept_pair_ids,
            args.activation_chunk_size,
            args.concept_methods,
            args.concept_layers,
        )
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
            loop_options = (
                None
                if args.disable_loop_detection
                else {
                    "ngram_size": args.loop_ngram_size,
                    "window_tokens": args.loop_window_tokens,
                    "unique_ratio_threshold": args.loop_unique_ratio_threshold,
                    "check_every": args.loop_check_every,
                    "consecutive_windows": args.loop_consecutive_windows,
                    "min_new_tokens": args.loop_min_new_tokens,
                    "extra_tokens": args.loop_extra_tokens,
                }
            )
            output, analysis, loop_detection = generate(
                model,
                tokenizer,
                formatted_prompt,
                scorer,
                name,
                example["id"],
                loop_options,
            )
            generation_seconds = time.perf_counter() - example_started
            if analysis is not None:
                generation_seconds -= analysis.analysis_seconds
            record = {
                "id": example["id"],
                "model": MODEL_ID,
                "thinking_enabled": True,
                "prompt_sha256": hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest(),
                "prompt": example["prompt"],
                "output": output,
                "reference": example["answer"],
                "correct": score(name, output, example["answer"]),
                "generation_seconds": generation_seconds,
                "loop_detected": loop_detection["detected"],
                "loop_forced_stop": loop_detection["forced_stop"],
                "loop_detection": loop_detection,
            }
            if analysis is not None:
                writer.add(analysis)
                record.update(
                    {
                        "concept_analysis_version": ANALYSIS_VERSION,
                        "concept_pair_ids": args.concept_pair_ids,
                        "concept_methods": list(args.concept_methods),
                        "concept_layers": list(args.concept_layers),
                        "reasoning_token_count": analysis.reasoning_token_count,
                        "reasoning_status": analysis.reasoning_status,
                        "reasoning_tokens": analysis.tokens,
                        "analysis_seconds": analysis.analysis_seconds,
                        "trace_dtype": "float16",
                        "activation_chunk_size": args.activation_chunk_size,
                    }
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{worker_label}] {name}: {n}/{len(examples)} | id={example['id']} "
                f"| correct={record['correct']} "
                f"| loop={record['loop_detected']} "
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
    command.extend(
        [
            "--loop-ngram-size",
            str(args.loop_ngram_size),
            "--loop-window-tokens",
            str(args.loop_window_tokens),
            "--loop-unique-ratio-threshold",
            str(args.loop_unique_ratio_threshold),
            "--loop-check-every",
            str(args.loop_check_every),
            "--loop-consecutive-windows",
            str(args.loop_consecutive_windows),
            "--loop-min-new-tokens",
            str(args.loop_min_new_tokens),
            "--loop-extra-tokens",
            str(args.loop_extra_tokens),
        ]
    )
    if args.disable_loop_detection:
        command.append("--disable-loop-detection")
    if args.concept_analysis:
        command.extend(
            [
                "--concept-analysis",
                "--activation-chunk-size",
                str(args.activation_chunk_size),
                "--concept-methods",
                ",".join(args.concept_methods),
                "--concept-layers",
                ",".join(map(str, args.concept_layers)),
                "--concept-pairs",
                ",".join(map(str, args.concept_pair_ids)),
            ]
        )
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
        compatible_records = matching_records(name, examples, canonical_records, require_analysis=args.concept_analysis, concept_pair_ids=args.concept_pair_ids, concept_methods=args.concept_methods, concept_layers=args.concept_layers)
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
        for record in matching_records(name, worker_examples, shard_records, require_analysis=args.concept_analysis, concept_pair_ids=args.concept_pair_ids, concept_methods=args.concept_methods, concept_layers=args.concept_layers):
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
    """Merge old and newly written worker scores without losing resumed examples."""
    columns = [
        "benchmark",
        "id",
        "method",
        "layer",
        "pair",
        "mean_cosine",
        "reasoning_tokens",
    ]
    key = ["benchmark", "id", "method", "layer", "pair"]
    destination = RESULTS / f"concept_scores-{name}.parquet"
    worker_paths = [
        RESULTS
        / f"concept_scores-{name}.worker-{worker:02d}-of-{args.num_workers:02d}.parquet"
        for worker in range(args.num_workers)
    ]
    sources = ([destination] if destination.exists() else []) + [
        path for path in worker_paths if path.exists()
    ]
    if not sources:
        print(
            f"No Parquet score rows were written for {name}; "
            "build_concept_report.py will recover them from the FP16 traces.",
            flush=True,
        )
        return

    expected_ids = {
        example["id"]
        for example in (
            load_benchmark(name)[: args.limit]
            if args.limit is not None
            else load_benchmark(name)
        )
    }
    frames = []
    for source in sources:
        frame = pd.read_parquet(source)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{source} is missing score columns: {missing}")
        frame = frame.loc[:, columns].copy()
        frame["id"] = frame["id"].astype(str)
        frame = frame[
            (frame["benchmark"].astype(str) == name)
            & frame["id"].isin(expected_ids)
            & frame["method"].astype(str).isin(args.concept_methods)
            & frame["layer"].astype(int).isin(args.concept_layers)
            & frame["pair"].astype(int).isin(args.concept_pair_ids)
        ]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        print(
            f"No compatible Parquet score rows were found for {name}; "
            "build_concept_report.py will recover them from the FP16 traces.",
            flush=True,
        )
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(key, keep="last").sort_values(key, kind="stable")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    merged.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)


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
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    for benchmark in benchmarks_for(args):
        run(benchmark, model, tokenizer, args)


if __name__ == "__main__":
    main()
