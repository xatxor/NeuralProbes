"""Small, streaming concept-vector scorer for Qwen3 evaluation outputs."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

VECTOR_REPO = "josephofthebread/Qwen3-8B-concept-vectors"
VECTOR_REVISION = "e15e1db9ca228c158aa4a372143922c8f66fb3c8"
LAYERS = (11, 14, 18, 22, 25)
METHODS = ("diff", "concept_centered", "antagonist_centered")
VERSION = 5


def _download(name: str) -> str:
    return hf_hub_download(VECTOR_REPO, name, revision=VECTOR_REVISION)


@dataclass
class AnalysisResult:
    reasoning_token_count: int
    reasoning_status: str
    tokens: list[str]
    color_scales: dict[str, float]
    score_rows: list[dict[str, Any]]
    highlight_rows: list[dict[str, Any]]
    analysis_seconds: float = 0.0


class ConceptScorer:
    """Hooks requested residual streams and scores them without saving activations."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device, trace_root: Path, highlights_per_sign: int = 3, pair_ids: list[int] | None = None, activation_chunk_size: int = 128, methods: tuple[str, ...] = METHODS) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trace_root = trace_root
        self.highlights_per_sign = highlights_per_sign
        self.activation_chunk_size = activation_chunk_size
        self.methods = methods
        if not self.methods or any(method not in METHODS for method in self.methods):
            raise ValueError(f"methods must be a non-empty subset of {METHODS}")
        if activation_chunk_size < 1:
            raise ValueError("activation_chunk_size must be at least 1")
        self.pairs = pd.read_parquet(_download("pairs.parquet"))
        if len(self.pairs) != 1036:
            raise ValueError(f"Expected 1036 concept pairs, found {len(self.pairs)}")
        self.pair_ids = list(range(len(self.pairs))) if pair_ids is None else sorted(set(pair_ids))
        if not self.pair_ids or min(self.pair_ids) < 0 or max(self.pair_ids) >= len(self.pairs):
            raise ValueError("Concept pair IDs must be between 0 and 1035")
        self.vectors = self._load_vectors()
        self.combined_vectors = {
            layer: torch.cat([self.vectors[method, layer][self.pair_ids] for method in self.methods], dim=0)
            for layer in LAYERS
        }
        self.captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
        self.handles: list[Any] = []
        self.streams: dict[tuple[str, int], Any] = {}
        self.temp_dir: Path | None = None
        self.captured_tokens = 0

    def _load_vectors(self) -> dict[tuple[str, int], torch.Tensor]:
        tensors: dict[tuple[str, int], torch.Tensor] = {}
        raw: dict[str, torch.Tensor] = {}
        for method in METHODS:
            path = _download(f"{method}.safetensors")
            with safe_open(path, framework="pt") as handle:
                metadata = handle.metadata() or {}
                tensor_name = next(iter(handle.keys()))
                if tuple(handle.get_slice(tensor_name).get_shape()) != (5, 1036, 4096):
                    raise ValueError(f"Unexpected {method} tensor shape")
                manifest = json.loads(metadata.get("manifest", "{}"))
                if manifest.get("layers") != list(LAYERS):
                    raise ValueError(f"Unexpected layers in {method}")
            raw[method] = load_file(path, device=str(self.device))[method].float()
        if not torch.allclose(raw["diff"], raw["concept_centered"] - raw["antagonist_centered"], atol=2e-5):
            raise ValueError("Vector files fail diff = concept_centered - antagonist_centered validation")
        for method, vector in raw.items():
            for index, layer in enumerate(LAYERS):
                tensors[method, layer] = F.normalize(vector[index], dim=-1)
        return tensors

    def _hook(self, layer: int) -> Callable[..., None]:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            residual = output[0] if isinstance(output, tuple) else output
            # Cached decoding has one input token. Prompt passes are deliberately ignored.
            if residual.ndim == 3 and residual.shape[1] == 1:
                self.captured[layer].append(residual[0, 0].detach())
                if layer == LAYERS[-1] and len(self.captured[layer]) >= self.activation_chunk_size:
                    self._flush_captured()

        return capture

    def begin(self, benchmark: str, example_id: str) -> None:
        self.captured = {layer: [] for layer in LAYERS}
        self.captured_tokens = 0
        self.temp_dir = self.trace_root / ".tmp" / benchmark / f"{example_id}-{os.getpid()}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.streams = {
            (method, layer): (self.temp_dir / f"{method}-L{layer}.bin").open("wb")
            for method in self.methods for layer in LAYERS
        }
        blocks = self.model.model.layers
        self.handles = [blocks[layer - 1].register_forward_hook(self._hook(layer)) for layer in LAYERS]

    def _flush_captured(self) -> None:
        count = min(len(values) for values in self.captured.values())
        if not count:
            return
        for layer in LAYERS:
            hidden = F.normalize(torch.stack(self.captured[layer][:count]).float(), dim=-1)
            projected = hidden @ self.combined_vectors[layer].T
            for method_index, method in enumerate(self.methods):
                scores = projected[:, method_index * len(self.pair_ids) : (method_index + 1) * len(self.pair_ids)]
                scores.detach().to(dtype=torch.float16).cpu().numpy().tofile(self.streams[method, layer])
            del self.captured[layer][:count]
        self.captured_tokens += count

    def _close_streams(self) -> None:
        for stream in self.streams.values():
            stream.close()
        self.streams = {}

    def _cleanup_temp(self) -> None:
        self._close_streams()
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.captured = {layer: [] for layer in LAYERS}

    def cancel(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._cleanup_temp()

    def finish(
        self,
        continuation: torch.Tensor,
        benchmark: str,
        example_id: str,
        write_highlights: Callable[[list[dict[str, Any]]], None] | None = None,
        span_mode: str = "thinking",
    ) -> AnalysisResult:
        analysis_started = time.perf_counter()
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._flush_captured()
        self._close_streams()
        # A cached forward on generated token i produces the residual for token i.
        available = min(len(continuation), self.captured_tokens)
        token_ids = continuation[:available].tolist()
        if span_mode == "thinking":
            start, end, status = thinking_span(self.tokenizer, continuation.tolist(), available)
        elif span_mode == "continuation":
            start, end, status = 0, available, "whole_continuation"
        else:
            raise ValueError("span_mode must be 'thinking' or 'continuation'")
        if start is None:
            self._cleanup_temp()
            return AnalysisResult(0, status, [], {}, [], [])
        if start == end:
            self._cleanup_temp()
            return AnalysisResult(0, "empty_thinking", [], {}, [], [])
        n_pairs = len(self.pair_ids)
        sums = {(method, layer): np.zeros(n_pairs, dtype=np.float32) for method in self.methods for layer in LAYERS}
        mins = {(method, layer): np.full(n_pairs, np.inf, dtype=np.float32) for method in self.methods for layer in LAYERS}
        maxs = {(method, layer): np.full(n_pairs, -np.inf, dtype=np.float32) for method in self.methods for layer in LAYERS}
        histograms = {(method, layer): np.zeros(1024, dtype=np.int64) for method in self.methods for layer in LAYERS}
        highlights: list[dict[str, Any]] = []
        trace_dir = self.trace_root / benchmark / str(example_id)
        trace_dir.mkdir(parents=True, exist_ok=True)
        for method in self.methods:
            for layer in LAYERS:
                source = np.memmap(self.temp_dir / f"{method}-L{layer}.bin", dtype=np.float16, mode="r", shape=(self.captured_tokens, n_pairs))
                target = np.lib.format.open_memmap(trace_dir / f"{method}-L{layer}.tmp.npy", mode="w+", dtype=np.float16, shape=(end - start, n_pairs))
                for offset in range(0, end - start, self.activation_chunk_size):
                    values = source[start + offset : min(end, start + offset + self.activation_chunk_size)]
                    target[offset : offset + len(values)] = values
                    scores = values.astype(np.float32)
                    key = method, layer
                    sums[key] += scores.sum(axis=0)
                    mins[key] = np.minimum(mins[key], scores.min(axis=0))
                    maxs[key] = np.maximum(maxs[key], scores.max(axis=0))
                    histograms[key] += np.bincount((np.abs(scores).clip(0, 1) * 1023).astype(np.int64).ravel(), minlength=1024)
                    if write_highlights is not None:
                        for local, row in enumerate(scores):
                            rows_for_token = self._highlights(torch.from_numpy(row), benchmark, example_id, method, layer, offset + local, token_ids[start + offset + local])
                            write_highlights(rows_for_token)
                del source, target
                (trace_dir / f"{method}-L{layer}.tmp.npy").replace(trace_dir / f"{method}-L{layer}.npy")
        rows = []
        for method in self.methods:
            for layer in LAYERS:
                key = method, layer
                means = (sums[key] / (end - start)).tolist()
                lo = mins[key].tolist()
                hi = maxs[key].tolist()
                for pair, mean, minimum, maximum in zip(self.pair_ids, means, lo, hi, strict=True):
                    rows.append({"benchmark": benchmark, "id": example_id, "method": method, "layer": layer, "pair": pair, "mean_cosine": mean, "min_cosine": minimum, "max_cosine": maximum, "reasoning_tokens": end - start})
        scales = {}
        for key, histogram in histograms.items():
            target = math.ceil(histogram.sum() * 0.99)
            scales[f"{key[0]}:{key[1]}"] = max(np.flatnonzero(histogram.cumsum() >= target)[0] / 1023, 1e-6)
        (trace_dir / "meta.tmp.json").write_text(json.dumps({"pair_ids": self.pair_ids, "tokens": [self.tokenizer.decode([token_ids[position]], skip_special_tokens=False) for position in range(start, end)], "dtype": "float16"}, ensure_ascii=False))
        (trace_dir / "meta.tmp.json").replace(trace_dir / "meta.json")
        self._cleanup_temp()
        return AnalysisResult(
            end - start,
            status,
            [self.tokenizer.decode([token_ids[position]], skip_special_tokens=False) for position in range(start, end)],
            scales,
            rows,
            highlights,
            time.perf_counter() - analysis_started,
        )

    def _highlights(self, scores: torch.Tensor, benchmark: str, example_id: str, method: str, layer: int, position: int, token_id: int) -> list[dict[str, Any]]:
        k = min(self.highlights_per_sign, scores.numel())
        positive, positive_ids = torch.topk(scores, k)
        negative, negative_ids = torch.topk(-scores, k)
        token = self.tokenizer.decode([token_id], skip_special_tokens=False)
        rows = []
        for polarity, values, ids in (("positive", positive, positive_ids), ("negative", -negative, negative_ids)):
            for value, pair in zip(values.cpu().tolist(), ids.cpu().tolist(), strict=True):
                rows.append({"benchmark": benchmark, "id": example_id, "method": method, "layer": layer, "token_index": position, "token": token, "pair": self.pair_ids[pair], "cosine": value, "polarity": polarity})
        return rows


def _subsequence(sequence: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for index in range(len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    return None


def thinking_span(tokenizer: Any, token_ids: list[int], available: int) -> tuple[int | None, int, str]:
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    start = _subsequence(token_ids[:available], opening)
    if start is None:
        return None, 0, "no_thinking_tags"
    start += len(opening)
    finish = _subsequence(token_ids[start:available], closing)
    if finish is None:
        return start, available, "unclosed_thinking"
    return start, start + finish, "closed_thinking"


class AnalysisWriter:
    """Append Parquet rows lazily; each worker owns its own files."""

    def __init__(self, root: Path, suffix: str) -> None:
        self.root = root
        self.suffix = suffix
        self.score_writer: pq.ParquetWriter | None = None
        self.highlight_writer: pq.ParquetWriter | None = None
        self.highlight_buffer: list[dict[str, Any]] = []

    def add(self, result: AnalysisResult) -> None:
        self.score_writer = self._write(self.score_writer, self.root / f"concept_scores{self.suffix}.parquet", result.score_rows)
        self.add_highlights(result.highlight_rows)

    def add_highlights(self, rows: list[dict[str, Any]]) -> None:
        self.highlight_buffer.extend(rows)
        if len(self.highlight_buffer) >= 50_000:
            self.highlight_writer = self._write(self.highlight_writer, self.root / f"token_highlights{self.suffix}.parquet", self.highlight_buffer)
            self.highlight_buffer.clear()

    @staticmethod
    def _write(writer: pq.ParquetWriter | None, path: Path, rows: list[dict[str, Any]]) -> pq.ParquetWriter | None:
        if not rows:
            return writer
        path.parent.mkdir(exist_ok=True)
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        return writer

    def close(self) -> None:
        self.add_highlights([])
        if self.highlight_buffer:
            self.highlight_writer = self._write(self.highlight_writer, self.root / f"token_highlights{self.suffix}.parquet", self.highlight_buffer)
            self.highlight_buffer.clear()
        for writer in (self.score_writer, self.highlight_writer):
            if writer is not None:
                writer.close()
