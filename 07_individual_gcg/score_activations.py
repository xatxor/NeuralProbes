"""Score saved responses along the steered concept without regenerating text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiment import CONCEPT, PAIR, all_responses, load_delta, load_model, render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        parser.error("worker-index must be in [0, num-workers)")
    assigned = all_responses(args.results)[args.worker_index :: args.num_workers]
    output = args.results / f"activations.worker-{args.worker_index:02d}.jsonl"
    done = {row["key"] for row in read_rows(output)}
    pending = [row for row in assigned if row["key"] not in done]
    if not pending:
        return
    model, tokenizer, device = load_model()
    vectors = {layer: F.normalize(load_delta(layer, device).float(), dim=0) for layer in (18, 25)}
    captures: dict[int, torch.Tensor] = {}
    handles = [
        model.model.layers[layer - 1].register_forward_hook(
            lambda _module, _inputs, value, layer=layer: captures.__setitem__(layer, value[0] if isinstance(value, tuple) else value)
        )
        for layer in vectors
    ]
    with output.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [tokenizer(render(tokenizer, row), add_special_tokens=False).input_ids for row in batch]
            continuations = [tokenizer(row["output"], add_special_tokens=False).input_ids for row in batch]
            encoded = tokenizer.pad(
                {"input_ids": [prompt + continuation for prompt, continuation in zip(prompts, continuations, strict=True)]},
                padding=True,
                return_tensors="pt",
            ).to(device)
            captures.clear()
            model.model(**encoded, use_cache=False)
            width = encoded.input_ids.shape[1]
            for index, row in enumerate(batch):
                padding = width - len(prompts[index]) - len(continuations[index])
                left, right = padding + len(prompts[index]), width
                record = {
                    key: row.get(key)
                    for key in ("key", "dataset", "id", "condition", "layer", "alpha", "intervention")
                } | {"pair": PAIR, "concept": CONCEPT, "response_tokens": right - left}
                for layer, vector in vectors.items():
                    hidden = F.normalize(captures[layer][index, left:right].float(), dim=-1)
                    record[f"L{layer}_mean_cosine"] = float((hidden @ vector).mean())
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"worker {args.worker_index}: {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    for handle in handles:
        handle.remove()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


if __name__ == "__main__":
    main()
