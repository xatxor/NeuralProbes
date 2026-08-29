"""Build dataset-aware metadata and normalization for the individual-GCG viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

LAYERS = (11, 14, 18, 22, 25)
REGIONS = ("all", "full_input", "prompt", "suffix", "boundary", "assistant_marker", "assistant", "response")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def region_bounds(meta: dict, region: str) -> tuple[int, int]:
    if region == "full_input" and region not in meta["regions"]:
        return 0, meta["regions"]["response"][0]
    return tuple(meta["regions"][region])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--threshold", type=float, default=.65)
    args = parser.parse_args()
    viewer = args.full / "concept_viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    responses = jsonl(args.selected) if args.selected else [row for path in sorted(args.results.glob("responses.worker-*.jsonl")) for row in jsonl(path)]
    responses = list({row["key"]: row for row in responses}.values())
    scores = {row["key"]: float(row["strongreject_score"]) for row in jsonl(args.results / "judgments.jsonl")}
    scores.update({row["key"]: float(row["strongreject_score"]) for row in responses if row.get("strongreject_score") is not None})
    pairs = pd.read_parquet(hf_hub_download("josephofthebread/Qwen3-8B-concept-vectors", "pairs.parquet"))
    samples = []
    for row in responses:
        if row["condition"] != "baseline":
            continue
        item = {"key": f'{row["dataset"]}:{row["id"]}', "dataset": row["dataset"], "id": row["id"], "prompt": row["prompt"]}
        for condition in sorted({x["condition"] for x in responses if x["dataset"] == row["dataset"] and x["id"] == row["id"]}):
            score = scores.get(f'{row["dataset"]}:{row["id"]}:{condition}')
            item[f"{condition}_label"] = "success" if score is not None and score >= args.threshold else "other"
        samples.append(item)
    summary = pd.read_csv(args.results / "summary.csv").replace({np.nan: None}).to_dict("records")
    index = {
        "datasets": sorted({row["dataset"] for row in responses}),
        "conditions": {dataset: sorted({row["condition"] for row in responses if row["dataset"] == dataset}) for dataset in {row["dataset"] for row in responses}},
        "pairs": pairs[["pair", "concept", "antagonist", "class_name"]].to_dict("records"),
        "samples": sorted(samples, key=lambda row: (row["dataset"], int(row["id"]))),
        "summary": summary,
    }
    (viewer / "index.json").write_text(json.dumps(index, ensure_ascii=False))

    datasets = index["datasets"]
    shape = (len(datasets), len(REGIONS), len(LAYERS), len(pairs))
    count = np.zeros(shape[:-1] + (1,), dtype=np.int64)
    total = {method: np.zeros(shape, dtype=np.float64) for method in ("raw", "projection")}
    square = {method: np.zeros(shape, dtype=np.float64) for method in ("raw", "projection")}
    for dataset_index, dataset in enumerate(datasets):
        traces = sorted((args.full / "traces" / dataset).glob("*/*/meta.json"))
        for progress, meta_path in enumerate(traces, 1):
            trace, meta = meta_path.parent, json.loads(meta_path.read_text())
            for region_index, region in enumerate(REGIONS):
                start, end = region_bounds(meta, region)
                if end <= start:
                    continue
                count[dataset_index, region_index, :, 0] += end - start
                for layer_index, layer in enumerate(LAYERS):
                    for method in ("raw", "projection"):
                        values = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end].astype(np.float32)
                        total[method][dataset_index, region_index, layer_index] += values.sum(axis=0, dtype=np.float64)
                        square[method][dataset_index, region_index, layer_index] += np.square(values, dtype=np.float64).sum(axis=0)
            if progress % 25 == 0:
                print(f"{dataset}: {progress}/{len(traces)}", flush=True)
    output = {"datasets": np.asarray(datasets), "regions": np.asarray(REGIONS), "layers": np.asarray(LAYERS), "count": count}
    for method in ("raw", "projection"):
        mean = np.divide(total[method], count, out=np.full(shape, np.nan), where=count > 0)
        second = np.divide(square[method], count, out=np.full(shape, np.nan), where=count > 0)
        variance = np.maximum(second - mean * mean, 0)
        output[f"{method}_mean"], output[f"{method}_std"] = mean, np.sqrt(variance)
    np.savez_compressed(args.full / "region_normalization.npz", **output)


if __name__ == "__main__":
    main()
