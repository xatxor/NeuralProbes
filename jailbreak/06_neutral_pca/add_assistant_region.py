"""Add the exact Qwen `assistant` boundary token to completed full traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = (11, 14, 18, 22, 25)
PAIRS = 1036


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    args = parser.parse_args()
    success = {(row["condition"], row["id"]): float(row["strongreject_score"]) >= 0.65 for row in jsonl(args.judgments)}
    sums, counts = defaultdict(lambda: np.zeros(PAIRS, dtype=np.float64)), defaultdict(int)
    for condition_dir in sorted((args.results / "traces").iterdir()):
        for trace in sorted(condition_dir.iterdir(), key=lambda path: int(path.name)):
            meta_path = trace / "meta.json"
            meta = json.loads(meta_path.read_text())
            start, end = meta["regions"]["boundary"]
            matches = [index for index in range(start, end) if meta["tokens"][index] == "assistant"]
            if len(matches) != 1:
                raise RuntimeError(f"{trace}: expected one assistant token, found {matches}")
            meta["regions"]["assistant"] = [matches[0], matches[0] + 1]
            meta_path.write_text(json.dumps(meta, ensure_ascii=False))
            scope = "success" if success[(condition_dir.name, trace.name)] else "other"
            for layer in LAYERS:
                for method in ("raw", "projection"):
                    values = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[matches[0]].astype(np.float32)
                    for group in ("all", scope):
                        key = condition_dir.name, layer, group, method
                        sums[key] += values
                        counts[key] += 1
    rows = []
    for (condition, layer, scope, method), values in sums.items():
        rows.extend({"condition": condition, "layer": layer, "region": "assistant", "scope": scope, "method": method, "pair": pair, "sum": float(value), "count": counts[condition, layer, scope, method], "mean_activation": float(value / counts[condition, layer, scope, method])} for pair, value in enumerate(values))
    path = args.results / "aggregate.parquet"
    current = pd.read_parquet(path)
    current = current[current.region != "assistant"]
    pd.concat([current, pd.DataFrame(rows)], ignore_index=True).to_parquet(path, index=False)


if __name__ == "__main__":
    main()
