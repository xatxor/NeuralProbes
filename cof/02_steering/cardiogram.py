"""Concept activation along the reasoning trace, for correct against incorrect answers.

Replays saved generations through the model without generating anything: each trace is
teacher-forced once, the residual stream is captured at the requested layers, projected
onto a concept direction, standardised over every reasoning token, and averaged into
equal relative-position bins. Correct and incorrect traces are averaged separately.

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

from steer import CONCEPTS, vector_manifest  # noqa: E402


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
    parser.add_argument("--max-tokens", type=int, default=None, help="Skip traces longer than this, to bound memory")
    return parser.parse_args()


def load_records(results: Path, benchmark: str, alpha: float) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(results.glob("steering*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
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
    sequence = torch.cat([prompt_ids, continuation_ids], dim=1).to(model.device)
    model(sequence)
    if capture.value is None:
        return None
    offset = prompt_ids.shape[1]
    hidden = capture.value[offset + start : offset + end]
    hidden = F.normalize(hidden.float(), dim=-1).to(dtype=directions.dtype)
    return (hidden @ directions.T).float().cpu().numpy().T


def binned(values: np.ndarray, bins: int) -> np.ndarray:
    """Average into equal relative-position bins; short traces repeat their nearest token."""
    positions = np.linspace(0, len(values), bins + 1).astype(int)
    out = np.empty(bins, dtype=float)
    for index in range(bins):
        low, high = positions[index], max(positions[index + 1], positions[index] + 1)
        out[index] = values[low:high].mean()
    return out


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

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    directions = load_directions(args.vector_dir, args.layer, pairs, model.device)
    capture = Capture(model, args.layer)

    per_trace: list[tuple[bool, np.ndarray]] = []
    pooled: list[np.ndarray] = []
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
                model, tokenizer, capture, record, instruction(args.benchmark, example["prompt"]), directions
            )
            if cosines is None:
                print(f"{index}/{len(records)} {record['key']}: no usable reasoning span, skipped", flush=True)
                continue
            per_trace.append((bool(record["correct"]), cosines))
            pooled.append(cosines)
            print(f"{index}/{len(records)} {record['key']} tokens={cosines.shape[1]} correct={record['correct']}", flush=True)
    finally:
        capture.close()

    if not per_trace:
        raise SystemExit("No traces produced usable reasoning spans")

    # Standardise within concept and layer over every reasoning token, as in the report.
    everything = np.concatenate(pooled, axis=1)
    mean = everything.mean(axis=1, keepdims=True)
    std = everything.std(axis=1, keepdims=True)
    std[std == 0] = 1.0

    groups = {True: [], False: []}
    for correct, cosines in per_trace:
        z = (cosines - mean) / std
        groups[correct].append(np.stack([binned(z[row], args.bins) for row in range(z.shape[0])]))

    x = (np.arange(args.bins) + 0.5) * 100 / args.bins
    columns = 2 if len(pairs) > 1 else 1
    rows = (len(pairs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 3.0 * rows), squeeze=False, sharex=True)
    for position, pair in enumerate(pairs):
        axis = axes[position // columns][position % columns]
        for correct, label, color in ((True, "Correct", "#3b76af"), (False, "Incorrect", "#d95f4c")):
            if not groups[correct]:
                continue
            stack = np.stack([item[position] for item in groups[correct]])
            axis.plot(x, stack.mean(axis=0), label=f"{label} (n={len(stack)})", color=color, linewidth=1.6)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(textwrap.fill(CONCEPTS.get(pair, str(pair)), 44), fontsize=10)
        axis.set_ylabel("Signed z-mean")
        axis.grid(alpha=0.25)
    for position in range(len(pairs), rows * columns):
        axes[position // columns][position % columns].remove()
    for column in range(columns):
        axes[rows - 1][column].set_xlabel("Relative position within reasoning, %")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(f"Concept activation along the reasoning trace (L{args.layer}, alpha={args.alpha:g})")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = out / f"cardiogram-L{args.layer}-a{args.alpha:g}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"{len(per_trace)} traces -> {path}")


if __name__ == "__main__":
    main()
