"""Serve the original GCG viewer with individual-GCG steering results."""

from __future__ import annotations

import argparse
import csv
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


SHARED_BOUNDARY_RESPONSE_RANKING = r'''
// Order by the weaker region first: a concept must rank highly in both.
const baseUpdateDistributionPairs=updateDistributionPairs;
const baseLoadDistributionRanking=loadDistributionRanking;
updateDistributionPairs=function(){
  if($('distributionRanking').value!=='boundary_response_shared')return baseUpdateDistributionPairs();
  let select=$('distributionPair'),old=select.value,q=$('distributionSearch').value.trim().toLowerCase(),byPair=new Map(distributionRanks.map(row=>[row.pair,row]));
  let rows=index.pairs.filter(x=>!q||`${x.pair} ${x.concept} ${x.antagonist}`.toLowerCase().includes(q)).sort((a,b)=>(byPair.get(a.pair)?.rank??Infinity)-(byPair.get(b.pair)?.rank??Infinity));
  select.innerHTML=rows.map(x=>{let row=byPair.get(x.pair),detail=row?`boundary #${row.boundary_rank} · response #${row.response_rank} · worst #${row.worst_rank}`:'';return `<option value="${x.pair}">${row?`#${row.rank} · ${detail} · `:''}${esc(x.concept)} ↔ ${esc(x.antagonist)}</option>`}).join('');
  if(rows.some(x=>String(x.pair)===old))select.value=old;
};
loadDistributionRanking=async function(){
  if($('distributionRanking').value!=='boundary_response_shared')return baseLoadDistributionRanking();
  let q=new URLSearchParams({layer:$('distributionLayer').value,metric:$('distributionScale').value,status:$('distributionStatus').value});
  $('distributionMeta').textContent='Calculating shared boundary/response ranking…';
  let [boundary,response]=await Promise.all(['boundary','response'].map(region=>fetch(`/api/distribution-ranking?${q}&region=${region}`).then(r=>r.json())));
  let ranks=rows=>new Map([...rows].sort((a,b)=>b.alpaca_like-a.alpaca_like).map((row,i)=>[row.pair,{rank:i+1,score:row.alpaca_like}]));
  let boundaryRanks=ranks(boundary),responseRanks=ranks(response);
  distributionRanks=[...boundaryRanks].filter(([pair])=>responseRanks.has(pair)).map(([pair,left])=>{let right=responseRanks.get(pair);return{pair,boundary_rank:left.rank,response_rank:right.rank,worst_rank:Math.max(left.rank,right.rank),mean_rank:(left.rank+right.rank)/2,boundary_score:left.score,response_score:right.score}}).sort((a,b)=>a.worst_rank-b.worst_rank||a.mean_rank-b.mean_rank).map((row,i)=>({...row,rank:i+1}));
  $('distributionMeta').textContent='Ordered by the weaker of the boundary and response GCG → Alpaca-like ranks.';
  updateDistributionPairs();renderDistributions();
};
let sharedOption=document.createElement('option');sharedOption.value='boundary_response_shared';sharedOption.textContent='Boundary + response shared';$('distributionRanking').append(sharedOption);
'''


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def pca2(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    return centered @ components[:2].T, variance[:2] / variance.sum(), components[:2]


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        query = urlparse(self.path)
        if query.path in {"/", "/viewer.html"}:
            return self._json(self.server.viewer, "text/html; charset=utf-8")  # type: ignore[attr-defined]
        if query.path == "/concept_auc_raw.json":
            return self._json(json.dumps(self._concept_auc()).encode())
        if query.path == "/api/individual-steering":
            return self._json(json.dumps(self.server.steering_summary).encode())  # type: ignore[attr-defined]
        if query.path == "/api/pca":
            args = parse_qs(query.query)
            layer, region = int(args.get("layer", ["25"])[0]), args.get("region", ["full_input"])[0]
            cache_key = layer, region
            if cache_key not in self.server.pca:  # type: ignore[attr-defined]
                matrix, points = [], []
                for dataset, condition in (("alpaca", "baseline"), ("advbench", "baseline"), ("advbench", "gcg"), ("advbench", "random")):
                    paths = self.server.trace_paths[dataset, condition]  # type: ignore[attr-defined]
                    if len(paths) != 100:
                        raise ValueError(f"Expected 100 {dataset}/{condition} traces, found {len(paths)}")
                    for path in paths:
                        meta = json.loads((path / "meta.json").read_text())
                        start, end = self._region(meta, region)
                        matrix.append(np.load(path / f"raw-L{layer}.npy", mmap_mode="r")[start:end].mean(axis=0, dtype=np.float32))
                        points.append({"dataset": dataset, "condition": condition, "id": path.name})
                coordinates, explained, components = pca2(np.asarray(matrix, dtype=np.float32))
                self.server.pca[cache_key] = {"shape": [len(matrix), len(matrix[0])], "explained": explained.tolist(), "pc1_loadings": components[0].tolist(), "pc2_loadings": components[1].tolist(), "points": [point | {"pc1": float(x), "pc2": float(y)} for point, (x, y) in zip(points, coordinates, strict=True)]}  # type: ignore[attr-defined]
            return self._json(json.dumps(self.server.pca[cache_key]).encode())  # type: ignore[attr-defined]
        if query.path == "/api/steering-example":
            args = parse_qs(query.query)
            key = (args.get("dataset", [""])[0], args.get("condition", [""])[0], args.get("layer", [""])[0], args.get("pair", [""])[0], args.get("alpha", [""])[0], args.get("id", [""])[0])
            return self._json(json.dumps(self.server.steering_responses.get(key, {}), ensure_ascii=False).encode())  # type: ignore[attr-defined]
        if query.path not in {"/api/trace", "/api/token", "/api/sample", "/api/aggregate", "/api/distribution", "/api/distribution-ranking"}:
            return super().do_GET()
        args = parse_qs(query.query)
        root = self.server.results  # type: ignore[attr-defined]
        dataset = args.get("dataset", ["advbench"])[0]
        if query.path == "/api/distribution-ranking":
            layer = int(args["layer"][0])
            metric, region = args.get("metric", ["z_score"])[0], args.get("region", ["response"])[0]
            status = args.get("status", ["all"])[0]
            cache_key = layer, metric, region, status
            if cache_key not in self.server.distribution_rankings:  # type: ignore[attr-defined]
                method = "projection" if metric in {"mean_projection", "pca_z_score"} else "raw"
                groups = {}
                for current_dataset, condition in (("alpaca", "baseline"), ("advbench", "baseline"), ("advbench", "random"), ("advbench", "gcg")):
                    if (current_dataset, condition) not in self.server.trace_paths:  # type: ignore[attr-defined]
                        continue
                    values = []
                    for trace in self.server.trace_paths[current_dataset, condition]:  # type: ignore[attr-defined]
                        if current_dataset == "advbench" and status != "all":
                            score = self.server.judgments.get(f"advbench:{trace.name}:gcg")  # type: ignore[attr-defined]
                            if (score is not None and score >= self.server.threshold) != (status == "success"):  # type: ignore[attr-defined]
                                continue
                        meta = json.loads((trace / "meta.json").read_text())
                        start, end = self._region(meta, region)
                        if end <= start:
                            continue
                        row = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end].mean(axis=0, dtype=np.float32)
                        if metric in {"z_score", "pca_z_score"}:
                            mean, std = self._region_stats(current_dataset, method, region, layer)
                            row = np.divide(row - mean, std, out=np.full(row.shape, np.nan), where=std > 1e-6)
                        values.append(row)
                    groups[current_dataset, condition] = np.asarray(values)
                required = [groups.get(key, np.empty(0)) for key in (("alpaca", "baseline"), ("advbench", "baseline"), ("advbench", "random"), ("advbench", "gcg"))]
                if not all(group.size for group in required):
                    ranking = []
                else:
                    q = np.linspace(0, 1, 101)
                    alpaca, baseline, random, gcg = required
                    distance = lambda left, right: np.trapezoid(np.abs(np.quantile(left, q, axis=0) - np.quantile(right, q, axis=0)), q, axis=0)
                    d_ga, d_gb, d_gr = distance(gcg, alpaca), distance(gcg, baseline), distance(gcg, random)
                    means = [np.nanmean(group, axis=0) for group in required]
                    ranking = [{
                        "pair": pair,
                        "alpaca_like": float((d_gb[pair] + d_gr[pair]) / 2 - d_ga[pair]),
                        "gcg_distinct": float(min(d_ga[pair], d_gb[pair], d_gr[pair])),
                        "d_gcg_alpaca": float(d_ga[pair]),
                        "d_gcg_baseline": float(d_gb[pair]),
                        "d_gcg_random": float(d_gr[pair]),
                        "mean_alpaca": float(means[0][pair]),
                        "mean_baseline": float(means[1][pair]),
                        "mean_random": float(means[2][pair]),
                        "mean_gcg": float(means[3][pair]),
                    } for pair in range(gcg.shape[1]) if np.isfinite(d_ga[pair] + d_gb[pair] + d_gr[pair])]
                self.server.distribution_rankings[cache_key] = ranking  # type: ignore[attr-defined]
            return self._json(json.dumps(self.server.distribution_rankings[cache_key]).encode())  # type: ignore[attr-defined]
        if query.path == "/api/distribution":
            layer, pair = int(args["layer"][0]), int(args["pair"][0])
            metric, region = args.get("metric", ["z_score"])[0], args.get("region", ["response"])[0]
            status = args.get("status", ["all"])[0]
            projected = metric in {"mean_projection", "pca_z_score"}
            method = "projection" if projected else "raw"
            rows = []
            datasets = list(self.server.conditions) if dataset == "both" else [dataset]  # type: ignore[attr-defined]
            for current_dataset in datasets:
                for condition in self.server.conditions[current_dataset]:  # type: ignore[attr-defined]
                    for trace in self.server.trace_paths[current_dataset, condition]:  # type: ignore[attr-defined]
                        meta = json.loads((trace / "meta.json").read_text())
                        start, end = self._region(meta, region)
                        if end <= start:
                            continue
                        value = float(np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end, pair].mean(dtype=np.float32))
                        if metric == "pca_z_score":
                            mean, std = self._region_stats(current_dataset, "projection", region, layer, pair)
                            value = (value - mean) / std if std > 1e-6 else np.nan
                        elif metric == "z_score":
                            mean, std = self._region_stats(current_dataset, "raw", region, layer, pair)
                            value = (value - mean) / std if std > 1e-6 else np.nan
                        if np.isfinite(value):
                            score = self.server.judgments.get(f"{current_dataset}:{trace.name}:{condition}")  # type: ignore[attr-defined]
                            cohort_score = self.server.judgments.get(f"{current_dataset}:{trace.name}:gcg")  # type: ignore[attr-defined]
                            if current_dataset != "alpaca" and status != "all" and (cohort_score is not None and cohort_score >= self.server.threshold) != (status == "success"):  # type: ignore[attr-defined]
                                continue
                            rows.append({
                                "dataset": current_dataset,
                                "condition": condition,
                                "series": f"{current_dataset} · {condition}" if len(datasets) > 1 else condition,
                                "id": trace.name,
                                "activation": float(value),
                                "strongreject_score": score,
                            })
            return self._json(json.dumps(rows).encode())
        if query.path == "/api/aggregate":
            layer = int(args["layer"][0])
            class_name = args.get("class", [""])[0].lower()
            class_search, search = args.get("class_search", [""])[0].lower(), args.get("q", [""])[0].lower()
            metric = args.get("metric", ["z_score"])[0]
            region = args.get("region", ["response"])[0]
            aggregate = self.server.full_aggregate  # type: ignore[attr-defined]
            rows = {}
            for condition in self.server.conditions[dataset]:  # type: ignore[attr-defined]
                for scope in ("all", "success", "other"):
                    key = f"{metric}:{dataset}:{condition}:{layer}:{region}:{scope}"
                    matches = []
                    source = aggregate.get(key) if (scope == "all" or self.server.threshold == .65) and not (region == "prompt" and self.server.prompt_includes_suffix) else None  # type: ignore[attr-defined]
                    if source is None:
                        source = self._aggregate_region(dataset, condition, layer, region, scope, metric)
                    for row in source or []:
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
        trace = self.server.full / "traces" / dataset / condition / sample_id  # type: ignore[attr-defined]
        meta = json.loads((trace / "meta.json").read_text())
        start, end = self._region(meta, region)
        if query.path == "/api/sample":
            response = self.server.responses[f"{dataset}:{sample_id}:{condition}"]  # type: ignore[attr-defined]
            projected = metric in {"mean_projection", "pca_z_score"}
            method = "projection" if projected else "raw"
            values = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")
            means = values[start:end].mean(axis=0, dtype=np.float32) if end > start else np.empty(0)
            rankings = []
            for pair, value in enumerate(means):
                row = {"pair": pair, metric: float(value)}
                if metric == "pca_z_score":
                    mean, std = self._region_stats(dataset, "projection", region, layer, pair)
                    row.update(mean_projection=float(value), pca_z_score=float((value - mean) / std) if std > 1e-6 else None)
                if not projected:
                    mean, std = self._region_stats(dataset, "raw", region, layer, pair)
                    row.update(mean_cosine=float(value), z_score=float((value - mean) / std) if std > 1e-6 else None)
                rankings.append(row)
            body = json.dumps({
                "response": response | {"attack_label": "success" if self.server.judgments.get(response["key"], 0) >= self.server.threshold else "other", "strongreject_score": self.server.judgments.get(response["key"]), "response_tokens": response["generated_tokens"]},  # type: ignore[attr-defined]
                "rankings": rankings,
            }).encode()
            return self._json(body)
        projected = metric in {"mean_projection", "pca_z_score"}
        values = np.load(trace / f"{'projection' if projected else 'raw'}-L{layer}.npy", mmap_mode="r")
        if query.path == "/api/trace":
            pair = int(args["pair"][0])
            raw = values[start:end, pair].astype(np.float32)
            if metric == "pca_z_score":
                mean, std = self._region_stats(dataset, "projection", region, layer, pair)
                z_values = (raw - mean) / std if std > 1e-6 else np.full(raw.shape, np.nan)
            elif projected:
                z_values = np.full(raw.shape, np.nan)
            else:
                mean, std = self._region_stats(dataset, "raw", region, layer, pair)
                z_values = (raw - mean) / std if std > 1e-6 else np.full(raw.shape, np.nan)
            body = json.dumps({"tokens": meta["tokens"][start:end], "values": raw.astype(float).tolist(), "z_values": [float(value) if np.isfinite(value) else None for value in z_values]}).encode()
        else:
            token = start + int(args["token"][0])
            if token >= end:
                return self._json(json.dumps({"error": "token outside selected region"}).encode())
            raw = values[token].astype(np.float32)
            if metric == "pca_z_score":
                mean, std = self._region_stats(dataset, "projection", region, layer)
                z_values = np.divide(raw - mean, std, out=np.full(raw.shape, np.nan), where=std > 1e-6)
            elif projected:
                z_values = np.full(raw.shape, np.nan)
            else:
                mean, std = self._region_stats(dataset, "raw", region, layer)
                z_values = np.divide(raw - mean, std, out=np.full(raw.shape, np.nan), where=std > 1e-6)
            body = json.dumps({"token": meta["tokens"][token], "values": raw.astype(float).tolist(), "z_values": [float(value) if np.isfinite(value) else None for value in z_values]}).encode()
        self._json(body)

    def _region_stats(self, dataset: str, method: str, region: str, layer: int, pair: int | None = None):
        if region not in self.server.region_index or (region == "prompt" and self.server.prompt_includes_suffix):  # type: ignore[attr-defined]
            key = dataset, method, region, layer
            if key not in self.server.dynamic_region_stats:  # type: ignore[attr-defined]
                values = []
                for condition in self.server.conditions[dataset]:  # type: ignore[attr-defined]
                    for trace in self.server.trace_paths[dataset, condition]:  # type: ignore[attr-defined]
                        meta = json.loads((trace / "meta.json").read_text())
                        start, end = self._region(meta, region)
                        values.append(np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end].astype(np.float32))
                merged = np.concatenate(values)
                self.server.dynamic_region_stats[key] = merged.mean(axis=0), merged.std(axis=0)  # type: ignore[attr-defined]
            mean, std = self.server.dynamic_region_stats[key]  # type: ignore[attr-defined]
            return (mean[pair], std[pair]) if pair is not None else (mean, std)
        index = self.server.region_index[region]  # type: ignore[attr-defined]
        layer_index = self.server.layer_index[layer]  # type: ignore[attr-defined]
        dataset_index = self.server.dataset_index[dataset]  # type: ignore[attr-defined]
        mean = self.server.region_mean[method][dataset_index, index, layer_index]  # type: ignore[attr-defined]
        std = self.server.region_std[method][dataset_index, index, layer_index]  # type: ignore[attr-defined]
        return (mean[pair], std[pair]) if pair is not None else (mean, std)

    def _region(self, meta: dict, region: str) -> tuple[int, int]:
        if region == "prompt" and self.server.prompt_includes_suffix:  # type: ignore[attr-defined]
            return meta["regions"]["prompt"][0], meta["regions"]["suffix"][1]
        if region == "full_input" and region not in meta["regions"]:
            return 0, meta["regions"]["response"][0]
        if region in meta["regions"]:
            return tuple(meta["regions"][region])
        if region == "assistant_marker":
            assistant, end = meta["regions"]["assistant"]
            if assistant == 0 or meta["tokens"][assistant - 1] != "<|im_start|>":
                raise RuntimeError("Qwen assistant marker was not found")
            return assistant - 1, end
        raise KeyError(region)

    def _aggregate_region(self, dataset: str, condition: str, layer: int, region: str, scope: str, metric: str):
        key = dataset, condition, layer, region, scope, metric
        if key in self.server.dynamic_aggregate:  # type: ignore[attr-defined]
            return self.server.dynamic_aggregate[key]  # type: ignore[attr-defined]
        method = "projection" if metric in {"mean_projection", "pca_z_score"} else "raw"
        rows = []
        for trace in self.server.trace_paths[dataset, condition]:  # type: ignore[attr-defined]
            if scope != "all":
                score = self.server.judgments.get(f"{dataset}:{trace.name}:{condition}")  # type: ignore[attr-defined]
                if (score is not None and score >= self.server.threshold) != (scope == "success"):  # type: ignore[attr-defined]
                    continue
            meta = json.loads((trace / "meta.json").read_text())
            start, end = self._region(meta, region)
            row = np.load(trace / f"{method}-L{layer}.npy", mmap_mode="r")[start:end].mean(axis=0, dtype=np.float32)
            if metric in {"z_score", "pca_z_score"}:
                mean, std = self._region_stats(dataset, method, region, layer)
                row = np.divide(row - mean, std, out=np.full(row.shape, np.nan), where=std > 1e-6)
            rows.append(row)
        values = np.nanmean(rows, axis=0) if rows else np.full(len(self.server.pairs), np.nan)  # type: ignore[attr-defined]
        result = [{"pair": pair, metric: float(value)} for pair, value in enumerate(values) if np.isfinite(value)]
        self.server.dynamic_aggregate[key] = result  # type: ignore[attr-defined]
        return result

    def _concept_auc(self) -> list[dict]:
        cached = self.server.concept_auc  # type: ignore[attr-defined]
        if cached is not None:
            return cached
        traces = self.server.trace_paths  # type: ignore[attr-defined]
        baseline, gcg = traces.get(("advbench", "baseline"), []), traces.get(("advbench", "gcg"), [])
        rows = []
        for layer in (11, 14, 18, 22, 25):
            for region in ("all", "full_input", "prompt", "suffix", "boundary", "assistant_marker", "assistant", "response"):
                def values(paths):
                    rows = []
                    for path in paths:
                        start, end = self._region(json.loads((path / "meta.json").read_text()), region)
                        if end > start:
                            rows.append(np.load(path / f"raw-L{layer}.npy", mmap_mode="r")[start:end].mean(axis=0, dtype=np.float32))
                    return np.asarray(rows)
                left, right = values(baseline), values(gcg)
                if not len(left) or not len(right):
                    continue
                combined = np.concatenate((left, right))
                order = np.argsort(combined, axis=0, kind="stable")
                ranks = np.empty_like(order, dtype=np.float32)
                ranks[order, np.arange(combined.shape[1])] = np.arange(len(combined))[:, None]
                auc = (ranks[len(left):].sum(axis=0) - len(right) * (len(right) - 1) / 2) / (len(left) * len(right))
                direction = np.where(right.mean(axis=0) >= left.mean(axis=0), "gcg > baseline", "baseline > gcg")
                rows.extend({"layer": layer, "region": region, "pair": pair, "separability_auc": float(value), "direction": str(direction[pair])} for pair, value in enumerate(auc))
        self.server.concept_auc = rows  # type: ignore[attr-defined]
        return rows

    def _json(self, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--individual-results", type=Path, required=True)
    parser.add_argument("--steering-results", type=Path, nargs="+")
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--hide-steering", action="store_true")
    parser.add_argument("--threshold", type=float, default=.65)
    parser.add_argument("--prompt-includes-suffix", action="store_true")
    parser.add_argument("--shared-boundary-response-ranking", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), lambda *a, **kw: Handler(*a, directory=str(args.results / "concept_viewer"), **kw))
    server.results = args.results  # type: ignore[attr-defined]
    viewer = (Path(__file__).parent / "viewer.html").read_text()
    viewer = viewer.replace("StrongREJECT score ≥ 0.65", f"StrongREJECT score ≥ {args.threshold:g}")
    if args.prompt_includes_suffix:
        viewer = viewer.replace("prompt:'User prompt'", "prompt:'Full user input (question + suffix)'")
    if args.hide_steering:
        viewer = viewer.replace("</head>", "<style>.tab[data-panel=steering],#steering{display:none!important}</style></head>")
    if args.shared_boundary_response_ranking:
        viewer = viewer.replace("document.addEventListener('change',event=>{", SHARED_BOUNDARY_RESPONSE_RANKING + "\ndocument.addEventListener('change',event=>{")
    server.viewer = viewer.encode()  # type: ignore[attr-defined]
    steering_roots = args.steering_results or [args.individual_results]
    server.steering_summary = [row for root in steering_roots for row in csv.DictReader((root / "summary.csv").open(newline=""))]  # type: ignore[attr-defined]
    steering_judgments = {row["key"]: row["strongreject_score"] for root in steering_roots for row in jsonl(root / "judgments.jsonl")}
    steering_responses = [row for root in steering_roots for path in sorted(root.glob("responses.worker-*.jsonl")) for row in jsonl(path)]
    server.steering_responses = {(row["dataset"], row["condition"], str(row["layer"]), str(row["concept_pair"]), f'{float(row["alpha"]):g}', str(row["id"])): row | {"strongreject_score": steering_judgments.get(row["key"])} for row in steering_responses}  # type: ignore[attr-defined]
    if args.selected:
        baseline = [row for row in jsonl(args.selected) if (row["dataset"], row["condition"]) in {("advbench", "gcg"), ("advbench", "baseline"), ("alpaca", "baseline") }]
        for summary in server.steering_summary:  # type: ignore[attr-defined]
            pair, layer, dataset, condition = str(summary["concept_pair"]), str(summary["layer"]), summary.get("dataset", "advbench"), summary.get("condition", "gcg")
            server.steering_responses.update({(dataset, condition, layer, pair, "0", str(row["id"])): row | {"alpha": 0, "layer": int(layer), "concept_pair": int(pair)} for row in baseline if (row["dataset"], row["condition"]) == (dataset, condition)})  # type: ignore[attr-defined]
    index = json.loads((args.results / "concept_viewer" / "index.json").read_text())
    server.pairs = {int(row["pair"]): row for row in index["pairs"]}  # type: ignore[attr-defined]
    server.conditions = index["conditions"]  # type: ignore[attr-defined]
    server.full = args.full_results  # type: ignore[attr-defined]
    server.trace_paths = {(dataset, condition): sorted((args.full_results / "traces" / dataset / condition).iterdir(), key=lambda path: int(path.name)) for dataset, conditions in server.conditions.items() for condition in conditions}  # type: ignore[attr-defined]
    selected = jsonl(args.selected) if args.selected else [row for path in sorted(args.individual_results.glob("responses.worker-*.jsonl")) for row in jsonl(path)]
    server.judgments = {row["key"]: float(row["strongreject_score"]) for line in (args.individual_results / "judgments.jsonl").read_text().splitlines() if line for row in [json.loads(line)]}  # type: ignore[attr-defined]
    selected = [row | {"key": f'{row["dataset"]}:{row["id"]}:{row["condition"]}'} for row in selected]
    server.judgments.update({row["key"]: float(row["strongreject_score"]) for row in selected if row.get("strongreject_score") is not None})  # type: ignore[attr-defined]
    server.responses = {row["key"]: row for row in selected}  # type: ignore[attr-defined]
    server.distribution_rankings = {}  # type: ignore[attr-defined]
    server.pca = {}  # type: ignore[attr-defined]
    server.concept_auc = None  # type: ignore[attr-defined]
    server.threshold = args.threshold  # type: ignore[attr-defined]
    server.prompt_includes_suffix = args.prompt_includes_suffix  # type: ignore[attr-defined]
    server.dynamic_region_stats = {}  # type: ignore[attr-defined]
    server.dynamic_aggregate = {}  # type: ignore[attr-defined]
    server.layer_index = {layer: index for index, layer in enumerate((11, 14, 18, 22, 25))}  # type: ignore[attr-defined]
    normalization = np.load(args.full_results / "region_normalization.npz")
    server.dataset_index = {str(dataset): index for index, dataset in enumerate(normalization["datasets"])}  # type: ignore[attr-defined]
    server.region_index = {str(region): index for index, region in enumerate(normalization["regions"])}  # type: ignore[attr-defined]
    server.region_mean = {method: normalization[f"{method}_mean"] for method in ("raw", "projection")}  # type: ignore[attr-defined]
    server.region_std = {method: normalization[f"{method}_std"] for method in ("raw", "projection")}  # type: ignore[attr-defined]
    aggregates = pd.read_parquet(args.full_results / "aggregate.parquet")
    server.full_aggregate = {}  # type: ignore[attr-defined]
    for keys, rows in aggregates.groupby(["dataset", "condition", "layer", "region", "scope", "method"], sort=False):
        dataset, condition, layer, region, scope, method = keys
        metric = "mean_projection" if method == "projection" else "mean_cosine"
        records = [{"pair": int(row.pair), metric: float(row.mean_activation)} for row in rows.itertuples()]
        server.full_aggregate[f"{metric}:{dataset}:{condition}:{layer}:{region}:{scope}"] = records
        if method == "raw":
            dataset_index, layer_index = server.dataset_index[str(dataset)], server.layer_index[int(layer)]  # type: ignore[attr-defined]
            if region not in server.region_index:  # type: ignore[attr-defined]
                continue
            region_index = server.region_index[region]  # type: ignore[attr-defined]
            server.full_aggregate[f"z_score:{dataset}:{condition}:{layer}:{region}:{scope}"] = [
                {
                    "pair": row["pair"],
                    "z_score": (
                        float((row["mean_cosine"] - server.region_mean["raw"][dataset_index, region_index, layer_index, row["pair"]]) / server.region_std["raw"][dataset_index, region_index, layer_index, row["pair"]])  # type: ignore[attr-defined]
                        if server.region_std["raw"][dataset_index, region_index, layer_index, row["pair"]] > 1e-6 else None  # type: ignore[attr-defined]
                    ),
                    "mean_cosine": row["mean_cosine"],
                }
                for row in records
            ]
        else:
            dataset_index, layer_index = server.dataset_index[str(dataset)], server.layer_index[int(layer)]  # type: ignore[attr-defined]
            if region not in server.region_index:  # type: ignore[attr-defined]
                continue
            region_index = server.region_index[region]  # type: ignore[attr-defined]
            server.full_aggregate[f"pca_z_score:{dataset}:{condition}:{layer}:{region}:{scope}"] = [
                {
                    "pair": row["pair"],
                    "pca_z_score": (
                        float((row["mean_projection"] - server.region_mean["projection"][dataset_index, region_index, layer_index, row["pair"]]) / server.region_std["projection"][dataset_index, region_index, layer_index, row["pair"]])  # type: ignore[attr-defined]
                        if server.region_std["projection"][dataset_index, region_index, layer_index, row["pair"]] > 1e-6 else None  # type: ignore[attr-defined]
                    ),
                    "mean_projection": row["mean_projection"],
                }
                for row in records
            ]
    print(f"http://localhost:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
