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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "01_eval"))

from concept_analysis import LAYERS, VECTOR_REPO, VECTOR_REVISION, thinking_span  # noqa: E402
from evaluate import MODEL_ID, instruction, load_benchmark, score, visible_gpu_ids  # noqa: E402

RESULTS = ROOT / "results"
STEERING_VERSION = 1
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
}
ALPHAS = (-0.1, -0.05, 0.0, 0.05, 0.1)


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
    parser.add_argument("--concept-pairs", default=",".join(map(str, CONCEPTS)))
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    args = parser.parse_args()
    args.concept_pairs = list(dict.fromkeys(comma_values(args.concept_pairs, int)))
    args.layers = list(dict.fromkeys(comma_values(args.layers, int)))
    args.alphas = list(dict.fromkeys(comma_values(args.alphas, float)))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    if not set(args.concept_pairs) <= set(CONCEPTS):
        parser.error(f"--concept-pairs must be selected from {list(CONCEPTS)}")
    if not set(args.layers) <= set(LAYERS):
        parser.error(f"--layers must be selected from {list(LAYERS)}")
    if not set(args.alphas) <= set(ALPHAS):
        parser.error(f"--alphas must be selected from {list(ALPHAS)}")
    return args


def benchmark_names(name: str) -> tuple[str, ...]:
    return ("aime_2024", "math_500", "gpqa_diamond") if name == "all" else (name,)


def condition_specs(concepts: list[int], layers: list[int], alphas: list[float]) -> list[dict[str, Any]]:
    conditions = []
    if 0.0 in alphas:
        conditions.append({"pair": None, "concept": None, "layer": None, "alpha": 0.0})
    conditions.extend(
        {"pair": pair, "concept": CONCEPTS[pair], "layer": layer, "alpha": alpha}
        for pair in concepts
        for layer in layers
        for alpha in alphas
        if alpha != 0.0
    )
    return conditions


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions = condition_specs(args.concept_pairs, args.layers, args.alphas)
    tasks = []
    for benchmark in benchmark_names(args.benchmark):
        examples = load_benchmark(benchmark)
        if args.limit is not None:
            examples = examples[: args.limit]
        for example in examples:
            prompt = instruction(benchmark, example["prompt"])
            for condition in conditions:
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
        condition = "baseline"
    else:
        condition = f"pair-{task['pair']}:L{task['layer']}:a{alpha_label(task['alpha'])}"
    return f"{task['benchmark']}:{task['id']}:{condition}"


def prompt_hash(task: dict[str, Any]) -> str:
    return hashlib.sha256(task["prompt"].encode()).hexdigest()


def compatible(record: dict[str, Any], task: dict[str, Any]) -> bool:
    return (
        record.get("key") == task_key(task)
        and record.get("model") == MODEL_ID
        and record.get("dtype") == "float16"
        and record.get("steering_version") == STEERING_VERSION
        and record.get("vector_revision") == VECTOR_REVISION
        and record.get("prompt_sha256") == prompt_hash(task)
    )


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


def load_deltas(concepts: list[int], layers: list[int], device: torch.device) -> dict[tuple[int, int], torch.Tensor]:
    pairs = pd.read_parquet(hf_hub_download(VECTOR_REPO, "pairs.parquet", revision=VECTOR_REVISION)).set_index("pair")
    tensor = load_file(hf_hub_download(VECTOR_REPO, "diff.safetensors", revision=VECTOR_REVISION))["diff"]
    if tuple(tensor.shape) != (len(LAYERS), 1036, 4096):
        raise ValueError(f"Unexpected diff vector shape: {tuple(tensor.shape)}")
    deltas = {}
    for pair in concepts:
        if pairs.loc[pair, "concept"] != CONCEPTS[pair]:
            raise ValueError(f"Concept metadata mismatch for pair {pair}")
        for layer in layers:
            vector = tensor[LAYERS.index(layer), pair].float()
            residual_norm = float(pairs.loc[pair, f"L{layer}_diff_norm"] / pairs.loc[pair, f"L{layer}_rel_norm"])
            deltas[pair, layer] = (F.normalize(vector, dim=0) * residual_norm).to(device=device, dtype=torch.float16)
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


@torch.inference_mode()
def generate(model: Any, tokenizer: Any, steerer: Steerer, task: dict[str, Any]) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with steerer.apply(task["pair"], task["layer"], task["alpha"]):
        output = model.generate(
            **inputs,
            max_length=model.config.max_position_embeddings,
            do_sample=False,
        )
    generation_seconds = time.perf_counter() - started
    continuation = output[0, inputs.input_ids.shape[1] :]
    token_ids = continuation.tolist()
    start, end, reasoning_status = thinking_span(tokenizer, token_ids, len(token_ids))
    reasoning_tokens = 0 if start is None else end - start
    text = tokenizer.decode(continuation, skip_special_tokens=False)
    ended_with_eos = bool(token_ids) and token_ids[-1] in eos_ids(tokenizer)
    hit_context_limit = output.shape[1] >= model.config.max_position_embeddings and not ended_with_eos
    correct = score(task["benchmark"], text, task["answer"]) if reasoning_status == "closed_thinking" else False
    return {
        "key": task_key(task),
        "benchmark": task["benchmark"],
        "id": task["id"],
        "model": MODEL_ID,
        "dtype": "float16",
        "steering_version": STEERING_VERSION,
        "vector_repo": VECTOR_REPO,
        "vector_revision": VECTOR_REVISION,
        "vector_method": "diff",
        "concept_pair": task["pair"],
        "concept": task["concept"],
        "layer": task["layer"],
        "alpha": task["alpha"],
        "prompt_sha256": prompt_hash(task),
        "output": text,
        "reference": task["answer"],
        "correct": correct,
        "generated_token_count": len(token_ids),
        "reasoning_token_count": reasoning_tokens,
        "reasoning_status": reasoning_status,
        "hit_context_limit": hit_context_limit,
        "generation_seconds": generation_seconds,
    }


def run_worker(args: argparse.Namespace) -> None:
    tasks = build_tasks(args)
    if args.worker_index is not None:
        tasks = tasks[args.worker_index :: args.num_workers]
    path = result_path(args)
    tasks_by_key = {task_key(task): task for task in tasks}
    completed = {
        record["key"]
        for record in iter_records(path)
        if record.get("key") in tasks_by_key and compatible(record, tasks_by_key[record["key"]])
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
        model.device,
    ) if nonzero else {}
    steerer = Steerer(model, deltas)
    RESULTS.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, 1):
            record = generate(model, tokenizer, steerer, task)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {record['key']} tokens={record['reasoning_token_count']} "
                f"correct={record['correct']} seconds={record['generation_seconds']:.1f}",
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
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def seed_shards(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
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
                if target and compatible(record, target[1]):
                    destinations[target[0]].write(line)
    finally:
        for handle in destinations.values():
            handle.close()


def merge_shards(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    tasks_by_key = {task_key(task): task for task in tasks}
    seen = set()
    destination = RESULTS / "steering.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for worker in range(args.num_workers):
            shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker})
            with result_path(shard_args).open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    key = record.get("key")
                    if key in tasks_by_key and key not in seen and compatible(record, tasks_by_key[key]):
                        output.write(line)
                        seen.add(key)
    missing = [key for key in tasks_by_key if key not in seen]
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing)} steering tasks are missing: {missing[:10]}")
    temporary.replace(destination)


def launch_workers(args: argparse.Namespace) -> None:
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(f"Requested {args.num_workers} workers, but only {len(gpu_ids)} GPUs are visible")
    tasks = build_tasks(args)
    RESULTS.mkdir(exist_ok=True)
    seed_shards(args, tasks)
    processes = []
    for worker in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker]
        processes.append(subprocess.Popen(worker_command(args, worker), env=environment))
    failures = [(worker, process.wait()) for worker, process in enumerate(processes)]
    failures = [(worker, code) for worker, code in failures if code]
    if failures:
        raise RuntimeError(f"Steering workers failed: {failures}")
    merge_shards(args, tasks)


def main() -> None:
    args = parse_args()
    if args.worker_index is None and args.num_workers > 1:
        launch_workers(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
