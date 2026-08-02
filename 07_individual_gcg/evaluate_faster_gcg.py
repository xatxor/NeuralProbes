"""Evaluate official Faster-GCG checkpoints plus clean and random controls."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiment import generate_batch, load_model, random_suffix  # noqa: E402


def read_pkl(path: Path) -> list[dict]:
    with path.open("rb") as stream:
        return pickle.load(stream)


def tasks(rows: list[dict], tokenizer: object, seed: int, alpaca: list[dict] | None = None) -> list[dict]:
    random = random_suffix(tokenizer, rows[0]["prompt"], rows[0]["target"], 20, seed)
    output = []
    for row in rows:
        base = {"dataset": "advbench", "id": str(row["index"]), "prompt": row["prompt"], "target": row["target"]}
        output += [base | {"key": f"advbench:{row['index']}:baseline", "condition": "baseline", "suffix": ""}]
        output += [base | {"key": f"advbench:{row['index']}:random", "condition": "random", "suffix": random}]
        candidates = sorted(zip(row["losses"], row["strings"]), key=lambda pair: pair[0])[:10]
        output += [base | {"key": f"advbench:{row['index']}:gcg:{candidate_index}", "condition": "gcg", "suffix": suffix, "candidate_index": candidate_index, "candidate_loss": loss} for candidate_index, (loss, suffix) in enumerate(candidates)]
    output += [
        {"key": f"alpaca:{row['id']}:baseline", "dataset": "alpaca", "id": str(row["id"]), "prompt": row["prompt"], "target": None, "condition": "baseline", "suffix": ""}
        for row in alpaca or []
    ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpaca-samples", type=Path)
    args = parser.parse_args()
    model, tokenizer, _device = load_model()
    alpaca = json.loads(args.alpaca_samples.read_text())["alpaca"] if args.alpaca_samples else []
    assigned = tasks(read_pkl(args.train), tokenizer, args.seed, alpaca)[args.worker_index :: args.workers]
    path = args.output / f"responses.worker-{args.worker_index:02d}.jsonl"
    done = {json.loads(line)["key"] for line in path.read_text().splitlines()} if path.exists() else set()
    pending = [row for row in assigned if row["key"] not in done]
    with path.open("a") as stream:
        for start in range(0, len(pending), 8):
            for row in generate_batch(model, tokenizer, pending[start : start + 8], args.max_new_tokens):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"worker {args.worker_index}: {min(start + 8, len(pending))}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
