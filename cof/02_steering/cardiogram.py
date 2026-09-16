"""Concept activation along the reasoning trace, for correct against incorrect answers.

Replays saved generations through the model without generating anything: each trace is
teacher-forced once, the residual stream is captured at the requested layers, projected
onto a concept direction, standardised over every reasoning token, and averaged into
equal relative-position bins. Correct and incorrect traces are averaged separately.

Every concept is scored in the same pass, and the per-trace scores are saved, so the
ranking of concepts by correct-against-incorrect separation (plot_concepts.py) needs no
second run through the model.

Runs on the saved steering.jsonl, so it needs a GPU and the model, but no regeneration.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "vika" / "01_eval"))

from concept_analysis import thinking_span  # noqa: E402
from evaluate import MODEL_ID, instruction, load_benchmark  # noqa: E402

from steer import CONCEPTS, STEERING_VERSION, vector_manifest  # noqa: E402

BOOTSTRAP_SAMPLES = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Where to write figures (default: --results)")
    parser.add_argument("--benchmark", default="gpqa_diamond")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--concept-pairs", default=None, help="Comma-separated pair IDs; default: every CoT concept")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.0, help="Which steering condition to read; 0 is the baseline")
    parser.add_argument("--max-tokens", type=int, default=None, help="Skip traces longer than this")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1024,
        help="How many tokens go through the model at once. Lower it if the card still runs out.",
    )
    return parser.parse_args()


def load_records(results: Path, benchmark: str, alpha: float) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(results.glob("steering*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A job killed mid-write can leave a truncated last line.
                    continue
                # An older pilot in the same folder reuses the baseline keys, so records from
                # other versions of steer.py are dropped before de-duplication.
                if record.get("steering_version") == STEERING_VERSION:
                    records[record["key"]] = record
    rows = [
        row for row in records.values()
        if row["benchmark"] == benchmark
        and row["alpha"] == alpha
        and row["reasoning_status"] == "closed_thinking"
    ]
    if not rows:
        raise SystemExit(f"No closed-thinking records at alpha={alpha} for {benchmark}")
    # One trace per question: repeated baselines are deterministic, so the first is enough.
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["key"]):
        unique.setdefault(row["id"], row)
    return list(unique.values())


def load_directions(vector_dir: Path, layer: int, pairs: list[int], device: torch.device) -> torch.Tensor:
    manifest = vector_manifest(vector_dir)
    if layer >= manifest["layers"]:
        raise SystemExit(f"Layer {layer} not in {vector_dir} (0..{manifest['layers'] - 1})")
    table = pd.read_parquet(vector_dir / "pairs.parquet").set_index("pair_id")
    tensor = load_file(vector_dir / "diff.safetensors")["diff"]
    for pair in pairs:
        if pair in CONCEPTS and table.loc[pair, "concept"] != CONCEPTS[pair]:
            raise SystemExit(f"Concept metadata mismatch for pair {pair}")
    stacked = torch.stack([tensor[layer, pair].float() for pair in pairs])
    return F.normalize(stacked, dim=-1).to(device=device, dtype=torch.float16)


class Capture:
    """Collects the residual stream of one block for a whole teacher-forced sequence."""

    def __init__(self, model: Any, layer: int) -> None:
        self.value: torch.Tensor | None = None
        self.handle = model.model.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        residual = output[0] if isinstance(output, tuple) else output
        self.value = residual[0].detach()

    def close(self) -> None:
        self.handle.remove()


@torch.inference_mode()
def trace_cosines(
    model: Any,
    tokenizer: Any,
    capture: Capture,
    record: dict[str, Any],
    prompt: str,
    directions: torch.Tensor,
    chunk_tokens: int,
) -> np.ndarray | None:
    """Per-reasoning-token cosine against each direction, shaped (pairs, reasoning tokens)."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_ids = tokenizer([rendered], return_tensors="pt").input_ids
    continuation_ids = tokenizer(record["output"], add_special_tokens=False, return_tensors="pt").input_ids
    token_ids = continuation_ids[0].tolist()
    start, end, status = thinking_span(tokenizer, token_ids, len(token_ids))
    if status != "closed_thinking" or start is None or end <= start:
        return None
    # Nothing after the reasoning span is needed, and the answer text can run long.
    sequence = torch.cat([prompt_ids, continuation_ids[:, :end]], dim=1).to(model.device)
    offset = prompt_ids.shape[1]

    # The trace goes through in chunks, carrying the KV cache: attention over many thousands
    # of tokens at once allocates more than the card has. Only the base model is called, so
    # no logits are built over the vocabulary at every position.
    captured, cache = [], None
    for position in range(0, sequence.shape[1], chunk_tokens):
        output = model.model(
            sequence[:, position : position + chunk_tokens], past_key_values=cache, use_cache=True
        )
        cache = output.past_key_values
        if capture.value is None:
            return None
        captured.append(capture.value)
    hidden = torch.cat(captured, dim=0)[offset + start : offset + end]
    hidden = F.normalize(hidden.float(), dim=-1).to(dtype=directions.dtype)
    return (hidden @ directions.T).float().cpu().numpy().T


def binned_all(values: np.ndarray, bins: int) -> np.ndarray:
    """Average every concept into equal relative-position bins, shaped (pairs, bins).

    Short traces repeat their nearest token, as before. With a thousand concepts a
    per-concept loop is what costs, so every bin is reduced in one pass.
    """
    edges = np.linspace(0, values.shape[1], bins + 1).astype(int)
    starts = np.minimum(edges[:-1], values.shape[1] - 1)
    counts = np.maximum(np.diff(edges), 1)
    return np.add.reduceat(values, starts, axis=1) / counts


def main() -> None:
    args = parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    pairs = (
        [int(value) for value in args.concept_pairs.split(",") if value.strip()]
        if args.concept_pairs
        else sorted(CONCEPTS)
    )

    records = load_records(args.results, args.benchmark, args.alpha)
    examples = {example["id"]: example for example in load_benchmark(args.benchmark)}

    # Every concept is scored in the same pass: one wider matrix multiply per trace, and the
    # saved scores then answer which concepts separate correct reasoning from incorrect.
    scored = [int(value) for value in pd.read_parquet(args.vector_dir / "pairs.parquet")["pair_id"]]
    missing = sorted(set(pairs) - set(scored))
    if missing:
        raise SystemExit(f"Pairs not in the vector table: {missing}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    directions = load_directions(args.vector_dir, args.layer, scored, model.device)
    capture = Capture(model, args.layer)

    per_trace: list[dict[str, Any]] = []
    token_sum = np.zeros(len(scored), dtype=np.float64)
    token_square = np.zeros(len(scored), dtype=np.float64)
    token_count = 0
    try:
        for index, record in enumerate(records, 1):
            if args.max_tokens and record["generated_token_count"] > args.max_tokens:
                print(f"{index}/{len(records)} {record['key']}: skipped, too long", flush=True)
                continue
            example = examples.get(record["id"])
            if example is None:
                print(f"{index}/{len(records)} {record['key']}: question not in benchmark, skipped", flush=True)
                continue
            cosines = trace_cosines(
                model,
                tokenizer,
                capture,
                record,
                instruction(args.benchmark, example["prompt"]),
                directions,
                args.chunk_tokens,
            )
            if cosines is None:
                print(f"{index}/{len(records)} {record['key']}: no usable reasoning span, skipped", flush=True)
                continue
            token_sum += cosines.sum(axis=1, dtype=np.float64)
            token_square += np.square(cosines, dtype=np.float64).sum(axis=1)
            token_count += cosines.shape[1]
            per_trace.append(
                {
                    "id": record["id"],
                    "correct": bool(record["correct"]),
                    "reasoning_tokens": cosines.shape[1],
                    # The per-trace mean is the statistic the previous phase ranked concepts by.
                    "mean_cosine": cosines.mean(axis=1),
                    "curve": binned_all(cosines, args.bins),
                }
            )
            print(
                f"{index}/{len(records)} {record['key']} tokens={cosines.shape[1]} correct={record['correct']}",
                flush=True,
            )
    finally:
        capture.close()

    if not per_trace:
        raise SystemExit("No traces produced usable reasoning spans")

    # Standardise within concept and layer over every reasoning token, as in the report.
    token_mean = token_sum / token_count
    token_std = np.sqrt(np.maximum(token_square / token_count - token_mean**2, 0.0))
    token_std[token_std == 0] = 1.0

    # Saved so that both figures can be rebuilt, on any concept, without the model.
    scores = out / f"concept-scores-L{args.layer}-a{args.alpha:g}.npz"
    np.savez_compressed(
        scores,
        pair_ids=np.asarray(scored, dtype=np.int32),
        ids=np.asarray([item["id"] for item in per_trace]),
        correct=np.asarray([item["correct"] for item in per_trace]),
        reasoning_tokens=np.asarray([item["reasoning_tokens"] for item in per_trace], dtype=np.int32),
        mean_cosine=np.stack([item["mean_cosine"] for item in per_trace]).astype(np.float32),
        binned_cosine=np.stack([item["curve"] for item in per_trace]).astype(np.float16),
        token_mean=token_mean.astype(np.float32),
        token_std=token_std.astype(np.float32),
        benchmark=args.benchmark,
        layer=args.layer,
        alpha=args.alpha,
    )

    rng = np.random.default_rng(20260916)
    position = {pair: index for index, pair in enumerate(scored)}
    groups: dict[bool, list[np.ndarray]] = {True: [], False: []}
    for item in per_trace:
        groups[item["correct"]].append((item["curve"] - token_mean[:, None]) / token_std[:, None])

    x = (np.arange(args.bins) + 0.5) * 100 / args.bins
    columns = 2 if len(pairs) > 1 else 1
    rows = (len(pairs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 3.0 * rows), squeeze=False, sharex=True)
    for index, pair in enumerate(pairs):
        axis = axes[index // columns][index % columns]
        for correct, label, color in ((True, "Correct", "#3b76af"), (False, "Incorrect", "#d95f4c")):
            if not groups[correct]:
                continue
            stack = np.stack([item[position[pair]] for item in groups[correct]])
            axis.plot(x, stack.mean(axis=0), label=f"{label} (n={len(stack)})", color=color, linewidth=1.6)
            # Bootstrap over traces: without a band a wiggle cannot be told from a difference.
            draws = stack[rng.integers(0, len(stack), size=(BOOTSTRAP_SAMPLES, len(stack)))].mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975], axis=0)
            axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(textwrap.fill(CONCEPTS.get(pair, str(pair)), 44), fontsize=10)
        axis.set_ylabel("Signed z-mean")
        axis.grid(alpha=0.25)
    for index in range(len(pairs), rows * columns):
        axes[index // columns][index % columns].remove()
    for column in range(columns):
        axes[rows - 1][column].set_xlabel("Relative position within reasoning, %")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(f"Concept activation along the reasoning trace (L{args.layer}, alpha={args.alpha:g})")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = out / f"cardiogram-L{args.layer}-a{args.alpha:g}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"{len(per_trace)} traces -> {path}, {scores}")


if __name__ == "__main__":
    main()
