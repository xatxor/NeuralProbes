"""Run single-concept, single-layer activation steering on Qwen3-8B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "vika" / "01_eval"))

from concept_analysis import thinking_span  # noqa: E402
from evaluate import (  # noqa: E402
    MODEL_ID,
    UniqueNGramLoopDetector,
    instruction,
    load_benchmark,
    score,
    visible_gpu_ids,
)

RESULTS = ROOT / "results"
STEERING_VERSION = 2
CONCEPTS = {
    367: "faithful chain-of-thought",
    960: "transparent chain-of-thought",
    357: "externalized scratchpad reasoning",
    909: "step-by-step reasoning",
    963: "transparent reasoning disclosure",
    598: "natural-language scaffold then formalize",
    908: "step explanation",
    253: "decomposing into subproblems",
    533: "keeping track of intermediate results",
    1013: "verifying intermediate steps",
    657: "planning the approach before calculating",
    902: "stating assumptions explicitly",
    703: "proof-style justification",
    146: "citing which given was used at each step",
    878: "slow thinking",
    459: "honest admission of not knowing",
    532: "joy",
    # random sanity-check concepts, unrelated to CoT process, seed=2026
    340: "eval-oblivious behavior",
    426: "good-faith red teamer",
    444: "healer",
    951: "training-time honesty about objectives",
    964: "treating benchmarks as ordinary tasks",
}
# The 16 CoT concepts plus the joy control. The random sanity-check pairs stay selectable
# through --concept-pairs but are out of the default grid.
DEFAULT_CONCEPT_PAIRS = (367, 960, 357, 909, 963, 598, 908, 253, 533, 1013, 657, 902, 703, 146, 878, 459, 532)
DEFAULT_LAYERS = (18,)
# Greedy decoding is near-deterministic: in the pilot four of five questions produced
# byte-identical baselines across three repeats, so one is enough.
DEFAULT_BASELINE_REPEATS = 1
# "concept" finishes one concept's whole alpha curve before starting the next;
# "alpha" covers every concept at one strength first.
DEFAULT_ORDER = "concept"
VALID_LAYERS = tuple(range(37))
# Alpha 3.5 and above is a non-termination regime rather than a stronger effect: in the
# pilot seven of ten generations at alpha 5 never closed their thinking block.
ALPHAS = (-3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0)


def comma_values(text: str, cast: Any) -> list[Any]:
    try:
        values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("aime_2024", "math_500", "gpqa_diamond", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--concept-pairs", default=",".join(map(str, DEFAULT_CONCEPT_PAIRS)))
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    parser.add_argument("--baseline-repeats", type=int, default=DEFAULT_BASELINE_REPEATS)
    parser.add_argument(
        "--order",
        choices=("concept", "alpha"),
        default=DEFAULT_ORDER,
        help="Which axis an interrupted run completes first: whole alpha curves per concept "
        "(concept), or every concept at one strength (alpha).",
    )
    parser.add_argument("--vector-dir", type=Path, required=True, help="Directory with manifest.json, diff.safetensors, pairs.parquet")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Generation budget per response. Strong steering often never terminates, and an "
        "unbounded run costs a full context window per generation.",
    )
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
    parser.add_argument(
        "--loop-extra-tokens",
        type=int,
        default=512,
        help="Tokens retained after loop detection so several repeated phrases remain analyzable.",
    )
    parser.add_argument("--loop-min-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.concept_pairs = list(dict.fromkeys(comma_values(args.concept_pairs, int)))
    args.layers = list(dict.fromkeys(comma_values(args.layers, int)))
    args.alphas = list(dict.fromkeys(comma_values(args.alphas, float)))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.baseline_repeats < 1:
        parser.error("--baseline-repeats must be at least 1")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    if not set(args.concept_pairs) <= set(CONCEPTS):
        parser.error(f"--concept-pairs must be selected from {list(CONCEPTS)}")
    if not set(args.layers) <= set(VALID_LAYERS):
        parser.error(f"--layers must be selected from {list(VALID_LAYERS)}")
    if not set(args.alphas) <= set(ALPHAS):
        parser.error(f"--alphas must be selected from {list(ALPHAS)}")
    if not args.vector_dir.is_dir():
        parser.error(f"--vector-dir {args.vector_dir} is not a directory")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def benchmark_names(name: str) -> tuple[str, ...]:
    return ("aime_2024", "math_500", "gpqa_diamond") if name == "all" else (name,)


def condition_specs(
    concepts: list[int],
    layers: list[int],
    alphas: list[float],
    baseline_repeats: int = DEFAULT_BASELINE_REPEATS,
    order: str = DEFAULT_ORDER,
) -> list[dict[str, Any]]:
    """Baselines first, then steered conditions in the order an interrupted run can use.

    With `order="concept"` alpha varies fastest, so each concept gets its complete
    dose-response curve before the next one starts. With `order="alpha"` concepts vary
    fastest, so every concept is covered at one strength before the next strength, which
    is what a cross-concept comparison at fixed alpha needs. Both respect the order the
    lists are given in, so --concept-pairs and --alphas set the priority.
    """
    conditions = []
    conditions.extend(
        {"pair": None, "concept": None, "layer": None, "alpha": 0.0, "baseline_repeat": repeat}
        for repeat in range(baseline_repeats)
    )
    pairs = (
        [(pair, alpha) for pair in concepts for alpha in alphas]
        if order == "concept"
        else [(pair, alpha) for alpha in alphas for pair in concepts]
    )
    conditions.extend(
        {"pair": pair, "concept": CONCEPTS[pair], "layer": layer, "alpha": alpha}
        for pair, alpha in pairs
        for layer in layers
        if alpha != 0.0
    )
    return conditions


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Tasks ordered condition-major: every question of one condition before the next.

    A run that is cut short then holds complete questions for the conditions it reached,
    which is what the accuracy deltas need, rather than every condition for a handful of
    questions. Baselines come first because every delta is measured against them.
    """
    conditions = condition_specs(args.concept_pairs, args.layers, args.alphas, args.baseline_repeats, args.order)
    tasks = []
    for benchmark in benchmark_names(args.benchmark):
        examples = load_benchmark(benchmark)
        if args.limit is not None:
            examples = examples[: args.limit]
        prompts = {example["id"]: instruction(benchmark, example["prompt"]) for example in examples}
        for condition in conditions:
            for example in examples:
                prompt = prompts[example["id"]]
                tasks.append(
                    {
                        "benchmark": benchmark,
                        "id": example["id"],
                        "prompt": prompt,
                        "answer": example["answer"],
                        **condition,
                    }
                )
    return tasks


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}"


def task_key(task: dict[str, Any]) -> str:
    if task["alpha"] == 0.0:
        repeat = task.get("baseline_repeat", 0)
        condition = "baseline" if repeat == 0 else f"baseline:repeat-{repeat}"
    else:
        condition = f"pair-{task['pair']}:L{task['layer']}:a{alpha_label(task['alpha'])}"
    return f"{task['benchmark']}:{task['id']}:{condition}"


def prompt_hash(task: dict[str, Any]) -> str:
    return hashlib.sha256(task["prompt"].encode()).hexdigest()


def compatible(record: dict[str, Any], task: dict[str, Any], capture_key: str, max_new_tokens: int) -> bool:
    if not (
        record.get("key") == task_key(task)
        and record.get("model") == MODEL_ID
        and record.get("dtype") == "float16"
        and record.get("steering_version") == STEERING_VERSION
        and record.get("vector_capture_key") == capture_key
        and record.get("prompt_sha256") == prompt_hash(task)
    ):
        return False
    if record.get("max_new_tokens") == max_new_tokens:
        return True
    # A generation that stopped on its own well inside the new budget is the same generation
    # this budget would produce, so a budget change does not invalidate it.
    generated = record.get("generated_token_count")
    stopped_on_its_own = not record.get("hit_context_limit") and not record.get("hit_token_budget", False)
    return stopped_on_its_own and isinstance(generated, int) and generated < max_new_tokens


def iter_records(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def result_path(args: argparse.Namespace) -> Path:
    if args.worker_index is None:
        return RESULTS / "steering.jsonl"
    return RESULTS / f"steering.worker-{args.worker_index:02d}-of-{args.num_workers:02d}.jsonl"


def vector_manifest(vector_dir: Path) -> dict[str, Any]:
    manifest = json.loads((vector_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL_ID or manifest["pairs"] != 1036 or manifest["hidden_size"] != 4096:
        raise ValueError(f"Unexpected vector manifest at {vector_dir}: {manifest}")
    return manifest


def load_deltas(concepts: list[int], layers: list[int], vector_dir: Path, device: torch.device) -> dict[tuple[int, int], torch.Tensor]:
    manifest = vector_manifest(vector_dir)
    pairs = pd.read_parquet(vector_dir / "pairs.parquet").set_index("pair_id")
    tensor = load_file(vector_dir / "diff.safetensors")["diff"]
    if tuple(tensor.shape) != (manifest["layers"], 1036, 4096):
        raise ValueError(f"Unexpected diff vector shape: {tuple(tensor.shape)}")
    deltas = {}
    for pair in concepts:
        if pairs.loc[pair, "concept"] != CONCEPTS[pair]:
            raise ValueError(f"Concept metadata mismatch for pair {pair}")
        for layer in layers:
            if layer >= manifest["layers"]:
                raise ValueError(f"Layer {layer} not present in {vector_dir} (0..{manifest['layers'] - 1})")
            deltas[pair, layer] = tensor[layer, pair].to(device=device, dtype=torch.float16)
    return deltas


class Steerer:
    def __init__(self, model: Any, deltas: dict[tuple[int, int], torch.Tensor]) -> None:
        self.model = model
        self.deltas = deltas

    @contextmanager
    def apply(self, pair: int | None, layer: int | None, alpha: float):
        if alpha == 0.0:
            yield
            return
        delta = self.deltas[pair, layer] * alpha

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            residual = output[0] if isinstance(output, tuple) else output
            steered = residual.clone()
            steered[:, -1, :] += delta
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        handle = self.model.model.layers[layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    return set(value if isinstance(value, list) else [value])


def loop_options(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.disable_loop_detection:
        return None
    return {
        "ngram_size": args.loop_ngram_size,
        "window_tokens": args.loop_window_tokens,
        "unique_ratio_threshold": args.loop_unique_ratio_threshold,
        "check_every": args.loop_check_every,
        "consecutive_windows": args.loop_consecutive_windows,
        "min_new_tokens": args.loop_min_new_tokens,
        "extra_tokens": args.loop_extra_tokens,
    }


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    steerer: Steerer,
    task: dict[str, Any],
    capture_key: str,
    options: dict[str, Any] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    detector = UniqueNGramLoopDetector(inputs.input_ids.shape[1], **options) if options is not None else None
    started = time.perf_counter()
    with steerer.apply(task["pair"], task["layer"], task["alpha"]):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=model.config.max_position_embeddings,
            do_sample=False,
            **({"stopping_criteria": StoppingCriteriaList([detector])} if detector is not None else {}),
        )
    generation_seconds = time.perf_counter() - started
    continuation = output[0, inputs.input_ids.shape[1] :]
    token_ids = continuation.tolist()
    start, end, reasoning_status = thinking_span(tokenizer, token_ids, len(token_ids))
    reasoning_tokens = 0 if start is None else end - start
    text = tokenizer.decode(continuation, skip_special_tokens=False)
    ended_with_eos = bool(token_ids) and token_ids[-1] in eos_ids(tokenizer)
    hit_context_limit = output.shape[1] >= model.config.max_position_embeddings and not ended_with_eos
    hit_token_budget = len(token_ids) >= max_new_tokens and not ended_with_eos
    correct = score(task["benchmark"], text, task["answer"]) if reasoning_status == "closed_thinking" else False
    loop_detection = detector.metadata(len(token_ids)) if detector is not None else {"enabled": False, "detected": False, "forced_stop": False}
    return {
        "key": task_key(task),
        "benchmark": task["benchmark"],
        "id": task["id"],
        "model": MODEL_ID,
        "dtype": "float16",
        "steering_version": STEERING_VERSION,
        "vector_capture_key": capture_key,
        "vector_method": "diff",
        "concept_pair": task["pair"],
        "concept": task["concept"],
        "layer": task["layer"],
        "alpha": task["alpha"],
        "baseline_repeat": task.get("baseline_repeat"),
        "prompt_sha256": prompt_hash(task),
        "output": text,
        "reference": task["answer"],
        "correct": correct,
        "generated_token_count": len(token_ids),
        "reasoning_token_count": reasoning_tokens,
        "reasoning_status": reasoning_status,
        "hit_context_limit": hit_context_limit,
        "hit_token_budget": hit_token_budget,
        "max_new_tokens": max_new_tokens,
        "loop_detected": loop_detection["detected"],
        "loop_forced_stop": loop_detection["forced_stop"],
        "loop_detection": loop_detection,
        "generation_seconds": generation_seconds,
    }


def run_worker(args: argparse.Namespace) -> None:
    capture_key = vector_manifest(args.vector_dir)["capture_merge_key"]
    tasks = build_tasks(args)
    if args.worker_index is not None:
        tasks = tasks[args.worker_index :: args.num_workers]
    path = result_path(args)
    tasks_by_key = {task_key(task): task for task in tasks}
    completed = {
        record["key"]
        for record in iter_records(path)
        if record.get("key") in tasks_by_key and compatible(record, tasks_by_key[record["key"]], capture_key, args.max_new_tokens)
    }
    pending = [task for task in tasks if task_key(task) not in completed]
    if not pending:
        print(f"{path}: all {len(tasks)} tasks already complete", flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nonzero = [task for task in pending if task["alpha"] != 0.0]
    deltas = load_deltas(
        sorted({task["pair"] for task in nonzero}),
        sorted({task["layer"] for task in nonzero}),
        args.vector_dir,
        model.device,
    ) if nonzero else {}
    steerer = Steerer(model, deltas)
    options = loop_options(args)
    RESULTS.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, 1):
            record = generate(model, tokenizer, steerer, task, capture_key, options, args.max_new_tokens)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {record['key']} tokens={record['reasoning_token_count']} "
                f"correct={record['correct']} loop={record['loop_detected']} "
                f"seconds={record['generation_seconds']:.1f}",
                flush=True,
            )


def worker_command(args: argparse.Namespace, worker: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--benchmark",
        args.benchmark,
        "--num-workers",
        str(args.num_workers),
        "--worker-index",
        str(worker),
        "--concept-pairs",
        ",".join(map(str, args.concept_pairs)),
        "--layers",
        ",".join(map(str, args.layers)),
        f"--alphas={','.join(map(str, args.alphas))}",
        "--baseline-repeats",
        str(args.baseline_repeats),
        "--order",
        args.order,
        "--vector-dir",
        str(args.vector_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
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
    if args.disable_loop_detection:
        command.append("--disable-loop-detection")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def seed_shards(args: argparse.Namespace, tasks: list[dict[str, Any]], capture_key: str) -> None:
    canonical = RESULTS / "steering.jsonl"
    if not canonical.exists():
        return
    missing_workers = []
    destinations = {}
    for worker in range(args.num_workers):
        shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker})
        path = result_path(shard_args)
        if not path.exists():
            missing_workers.append(worker)
            destinations[worker] = path.open("w", encoding="utf-8")
    if not missing_workers:
        return
    assigned = {
        task_key(task): (index % args.num_workers, task)
        for index, task in enumerate(tasks)
        if index % args.num_workers in missing_workers
    }
    try:
        with canonical.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                target = assigned.get(record.get("key"))
                if target and compatible(record, target[1], capture_key, args.max_new_tokens):
                    destinations[target[0]].write(line)
    finally:
        for handle in destinations.values():
            handle.close()


def merge_shards(args: argparse.Namespace, tasks: list[dict[str, Any]], capture_key: str) -> None:
    tasks_by_key = {task_key(task): task for task in tasks}
    seen = set()
    destination = RESULTS / "steering.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        if destination.exists():
            with destination.open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    if record.get("key") not in tasks_by_key:
                        output.write(line)
        for worker in range(args.num_workers):
            shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker})
            with result_path(shard_args).open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    key = record.get("key")
                    if key in tasks_by_key and key not in seen and compatible(record, tasks_by_key[key], capture_key, args.max_new_tokens):
                        output.write(line)
                        seen.add(key)
    missing = [key for key in tasks_by_key if key not in seen]
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing)} steering tasks are missing: {missing[:10]}")
    temporary.replace(destination)


def launch_workers(args: argparse.Namespace) -> None:
    capture_key = vector_manifest(args.vector_dir)["capture_merge_key"]
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(f"Requested {args.num_workers} workers, but only {len(gpu_ids)} GPUs are visible")
    tasks = build_tasks(args)
    RESULTS.mkdir(exist_ok=True)
    seed_shards(args, tasks, capture_key)
    processes = []
    for worker in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker]
        processes.append(subprocess.Popen(worker_command(args, worker), env=environment))
    failures = [(worker, process.wait()) for worker, process in enumerate(processes)]
    failures = [(worker, code) for worker, code in failures if code]
    if failures:
        raise RuntimeError(f"Steering workers failed: {failures}")
    merge_shards(args, tasks, capture_key)


def main() -> None:
    args = parse_args()
    if args.worker_index is None and args.num_workers > 1:
        launch_workers(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
