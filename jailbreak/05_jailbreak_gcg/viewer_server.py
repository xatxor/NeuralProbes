"""Serve the GCG viewer and fetch one memory-mapped trace column at a time."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        query = urlparse(self.path)
        if query.path not in {"/api/trace", "/api/token", "/api/sample", "/api/aggregate", "/api/distribution"}:
            return super().do_GET()
        args = parse_qs(query.query)
        root = self.server.results  # type: ignore[attr-defined]
        if query.path == "/api/distribution":
            layer, pair = int(args["layer"][0]), int(args["pair"][0])
            metric, region = args.get("metric", ["z_score"])[0], args.get("region", ["response"])[0]
            projected = metric in {"mean_projection", "pca_z_score"}
            method = "projection" if projected else "raw"
            rows = []
            for condition in self.server.conditions:  # type: ignore[attr-defined]
                for trace in self.server.trace_paths[condition]:  # type: ignore[attr-defined]
                    meta = json.loads((trace / "meta.json").read_text())
                    start, end = meta["regions"][region]
                    if end <= start:
                        continue
                    value = float(np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end, pair].mean(dtype=np.float32))
                    if metric == "pca_z_score":
                        mean, std = self._region_stats("projection", region, layer, pair)
                        value = (value - mean) / std if std > 1e-6 else np.nan
                    elif metric == "z_score":
                        mean, std = self._region_stats("raw", region, layer, pair)
                        value = (value - mean) / std if std > 1e-6 else np.nan
                    if np.isfinite(value):
                        rows.append({"condition": condition, "id": trace.name, "activation": float(value), "strongreject_score": self.server.judgments[condition, trace.name]})  # type: ignore[attr-defined]
            return self._json(json.dumps(rows).encode())
        if query.path == "/api/aggregate":
            layer = int(args["layer"][0])
            class_name = args.get("class", [""])[0].lower()
            class_search, search = args.get("class_search", [""])[0].lower(), args.get("q", [""])[0].lower()
            metric = args.get("metric", ["z_score"])[0]
            region = args.get("region", ["response"])[0]
            aggregate = self.server.full_aggregate  # type: ignore[attr-defined]
            rows = {}
            for condition in self.server.conditions:  # type: ignore[attr-defined]
                for scope in ("all", "success", "other"):
                    key = f"{metric}:{condition}:{layer}:{region}:{scope}"
                    matches = []
                    for row in aggregate.get(key, []):
                        pair = self.server.pairs[int(row["pair"])]  # type: ignore[attr-defined]
                        text = f'{row["pair"]} {pair["concept"]} {pair["antagonist"]} {pair["class_name"]}'.lower()
                        pair_class = pair["class_name"].lower()
                        if (not class_name or class_name == pair_class) and (not class_search or class_search in pair_class) and (not search or search in text):
                            matches.append(row)
                    ranked = sorted(matches, key=lambda row: row[metric] if row.get(metric) is not None else float("-inf"), reverse=True)
                    rows[f"{condition}:{layer}:{scope}"] = {"top": ranked[:20]}
            return self._json(json.dumps(rows).encode())
        condition, sample_id = args["condition"][0], args["id"][0]
        layer = int(args["layer"][0])
        metric = args.get("metric", ["z_score"])[0]
        region = args.get("region", ["response"])[0]
        trace = self.server.full / "traces" / condition / sample_id  # type: ignore[attr-defined]
        meta = json.loads((trace / "meta.json").read_text())
        start, end = meta["regions"][region]
        if query.path == "/api/sample":
            sample = json.loads((root / "concept_viewer" / f"sample-{sample_id}.json").read_text())
            projected = metric in {"mean_projection", "pca_z_score"}
            method = "projection" if projected else "raw"
            values = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")
            means = values[start:end].mean(axis=0, dtype=np.float32) if end > start else np.empty(0)
            rankings = []
            for pair, value in enumerate(means):
                row = {"pair": pair, metric: float(value)}
                if metric == "pca_z_score":
                    mean, std = self._region_stats("projection", region, layer, pair)
                    row.update(mean_projection=float(value), pca_z_score=float((value - mean) / std) if std > 1e-6 else None)
                if not projected:
                    mean, std = self._region_stats("raw", region, layer, pair)
                    row.update(mean_cosine=float(value), z_score=float((value - mean) / std) if std > 1e-6 else None)
                rankings.append(row)
            body = json.dumps({
                "response": next(row for row in sample["responses"] if row["condition"] == condition),
                "rankings": rankings,
            }).encode()
            return self._json(body)
        projected = metric in {"mean_projection", "pca_z_score"}
        values = np.load(trace / f"{'projection' if projected else 'raw'}-L{layer}.npy", mmap_mode="r")
        if query.path == "/api/trace":
            pair = int(args["pair"][0])
            raw = values[start:end, pair].astype(np.float32)
            if metric == "pca_z_score":
                mean, std = self._region_stats("projection", region, layer, pair)
                z_values = (raw - mean) / std if std > 1e-6 else np.full(raw.shape, np.nan)
            elif projected:
                z_values = np.full(raw.shape, np.nan)
            else:
                mean, std = self._region_stats("raw", region, layer, pair)
                z_values = (raw - mean) / std if std > 1e-6 else np.full(raw.shape, np.nan)
            body = json.dumps({"tokens": meta["tokens"][start:end], "values": raw.astype(float).tolist(), "z_values": [float(value) if np.isfinite(value) else None for value in z_values]}).encode()
        else:
            token = start + int(args["token"][0])
            if token >= end:
                return self._json(json.dumps({"error": "token outside selected region"}).encode())
            raw = values[token].astype(np.float32)
            if metric == "pca_z_score":
                mean, std = self._region_stats("projection", region, layer)
                z_values = np.divide(raw - mean, std, out=np.full(raw.shape, np.nan), where=std > 1e-6)
            elif projected:
                z_values = np.full(raw.shape, np.nan)
            else:
                mean, std = self._region_stats("raw", region, layer)
                z_values = np.divide(raw - mean, std, out=np.full(raw.shape, np.nan), where=std > 1e-6)
            body = json.dumps({"token": meta["tokens"][token], "values": raw.astype(float).tolist(), "z_values": [float(value) if np.isfinite(value) else None for value in z_values]}).encode()
        self._json(body)

    def _region_stats(self, method: str, region: str, layer: int, pair: int | None = None):
        if region not in self.server.region_index:  # type: ignore[attr-defined]
            return (np.nan, np.nan) if pair is not None else (np.full(1036, np.nan), np.full(1036, np.nan))
        index = self.server.region_index[region]  # type: ignore[attr-defined]
        layer_index = self.server.layer_index[layer]  # type: ignore[attr-defined]
        mean = self.server.region_mean[method][index, layer_index]  # type: ignore[attr-defined]
        std = self.server.region_std[method][index, layer_index]  # type: ignore[attr-defined]
        return (mean[pair], std[pair]) if pair is not None else (mean, std)

    def _json(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), lambda *a, **kw: Handler(*a, directory=str(args.results / "concept_viewer"), **kw))
    server.results = args.results  # type: ignore[attr-defined]
    index = json.loads((args.results / "concept_viewer" / "index.json").read_text())
    server.aggregate = json.loads((args.results / "concept_viewer" / "aggregate.json").read_text())  # type: ignore[attr-defined]
    server.pairs = {int(row["pair"]): row for row in index["pairs"]}  # type: ignore[attr-defined]
    server.conditions = index["conditions"]  # type: ignore[attr-defined]
    server.full = args.full_results  # type: ignore[attr-defined]
    server.trace_paths = {condition: sorted((args.full_results / "traces" / condition).iterdir(), key=lambda path: int(path.name)) for condition in server.conditions}  # type: ignore[attr-defined]
    server.judgments = {(row["condition"], row["id"]): float(row["strongreject_score"]) for line in (args.results / "judgments.jsonl").read_text().splitlines() if line for row in [json.loads(line)]}  # type: ignore[attr-defined]
    server.layer_index = {layer: index for index, layer in enumerate((11, 14, 18, 22, 25))}  # type: ignore[attr-defined]
    normalization = np.load(args.full_results / "advbench_region_normalization.npz")
    server.region_index = {str(region): index for index, region in enumerate(normalization["regions"])}  # type: ignore[attr-defined]
    server.region_mean = {method: normalization[f"{method}_mean"] for method in ("raw", "projection")}  # type: ignore[attr-defined]
    server.region_std = {method: normalization[f"{method}_std"] for method in ("raw", "projection")}  # type: ignore[attr-defined]
    aggregates = pd.read_parquet(args.full_results / "aggregate.parquet")
    server.full_aggregate = {}  # type: ignore[attr-defined]
    for keys, rows in aggregates.groupby(["condition", "layer", "region", "scope", "method"], sort=False):
        condition, layer, region, scope, method = keys
        metric = "mean_projection" if method == "projection" else "mean_cosine"
        records = [{"pair": int(row.pair), metric: float(row.mean_activation)} for row in rows.itertuples()]
        server.full_aggregate[f"{metric}:{condition}:{layer}:{region}:{scope}"] = records
        if method == "raw":
            layer_index = server.layer_index[int(layer)]  # type: ignore[attr-defined]
            if region not in server.region_index:  # type: ignore[attr-defined]
                continue
            region_index = server.region_index[region]  # type: ignore[attr-defined]
            server.full_aggregate[f"z_score:{condition}:{layer}:{region}:{scope}"] = [
                {
                    "pair": row["pair"],
                    "z_score": (
                        float((row["mean_cosine"] - server.region_mean["raw"][region_index, layer_index, row["pair"]]) / server.region_std["raw"][region_index, layer_index, row["pair"]])  # type: ignore[attr-defined]
                        if server.region_std["raw"][region_index, layer_index, row["pair"]] > 1e-6 else None  # type: ignore[attr-defined]
                    ),
                    "mean_cosine": row["mean_cosine"],
                }
                for row in records
            ]
        else:
            layer_index = server.layer_index[int(layer)]  # type: ignore[attr-defined]
            if region not in server.region_index:  # type: ignore[attr-defined]
                continue
            region_index = server.region_index[region]  # type: ignore[attr-defined]
            server.full_aggregate[f"pca_z_score:{condition}:{layer}:{region}:{scope}"] = [
                {
                    "pair": row["pair"],
                    "pca_z_score": (
                        float((row["mean_projection"] - server.region_mean["projection"][region_index, layer_index, row["pair"]]) / server.region_std["projection"][region_index, layer_index, row["pair"]])  # type: ignore[attr-defined]
                        if server.region_std["projection"][region_index, layer_index, row["pair"]] > 1e-6 else None  # type: ignore[attr-defined]
                    ),
                    "mean_projection": row["mean_projection"],
                }
                for row in records
            ]
    print(f"http://localhost:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
