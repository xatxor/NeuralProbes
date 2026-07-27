"""Small, streaming concept-vector scorer for Qwen3 evaluation outputs."""

from __future__ import annotations

import json
import math
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
VERSION = 2


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


class ConceptScorer:
    """Hooks requested residual streams and scores them without saving activations."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device, highlights_per_sign: int = 3) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.highlights_per_sign = highlights_per_sign
        self.pairs = pd.read_parquet(_download("pairs.parquet"))
        if len(self.pairs) != 1036:
            raise ValueError(f"Expected 1036 concept pairs, found {len(self.pairs)}")
        self.vectors = self._load_vectors()
        self.captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
        self.handles: list[Any] = []

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

        return capture

    def begin(self) -> None:
        self.captured = {layer: [] for layer in LAYERS}
        blocks = self.model.model.layers
        self.handles = [blocks[layer - 1].register_forward_hook(self._hook(layer)) for layer in LAYERS]

    def finish(
        self,
        continuation: torch.Tensor,
        benchmark: str,
        example_id: str,
        write_highlights: Callable[[list[dict[str, Any]]], None] | None = None,
        write_selected: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> AnalysisResult:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        # A cached forward on generated token i produces the residual for token i.
        available = min(len(continuation), *(len(items) for items in self.captured.values()))
        token_ids = continuation[:available].tolist()
        start, end, status = thinking_span(self.tokenizer, continuation.tolist(), available)
        if start is None:
            return AnalysisResult(0, status, [], {}, [], [])
        positions = list(range(start, end))
        if not positions:
            return AnalysisResult(0, "empty_thinking", [], {}, [], [])
        sums = {(method, layer): torch.zeros(len(self.pairs), device=self.device) for method in METHODS for layer in LAYERS}
        mins = {(method, layer): torch.full((len(self.pairs),), float("inf"), device=self.device) for method in METHODS for layer in LAYERS}
        maxs = {(method, layer): torch.full((len(self.pairs),), float("-inf"), device=self.device) for method in METHODS for layer in LAYERS}
        histograms = {(method, layer): torch.zeros(1024, dtype=torch.int64, device=self.device) for method in METHODS for layer in LAYERS}
        highlights: list[dict[str, Any]] = []
        for token_position in positions:
            for layer in LAYERS:
                hidden = F.normalize(self.captured[layer][token_position].float(), dim=-1)
                for method in METHODS:
                    scores = hidden @ self.vectors[method, layer].T
                    key = method, layer
                    sums[key] += scores
                    mins[key] = torch.minimum(mins[key], scores)
                    maxs[key] = torch.maximum(maxs[key], scores)
                    histograms[key].scatter_add_(0, (scores.abs().clamp(0, 1) * 1023).long(), torch.ones_like(scores, dtype=torch.int64))
                    rows_for_token = self._highlights(scores, benchmark, example_id, method, layer, token_position, token_ids[token_position])
                    if write_highlights is None:
                        highlights.extend(rows_for_token)
                    else:
                        write_highlights(rows_for_token)
        rows = []
        selected_pairs: dict[tuple[str, int], torch.Tensor] = {}
        for method in METHODS:
            for layer in LAYERS:
                key = method, layer
                means = (sums[key] / len(positions)).cpu().tolist()
                mean_tensor = sums[key] / len(positions)
                selected_pairs[key] = torch.cat((torch.topk(mean_tensor, 3).indices, torch.topk(-mean_tensor, 3).indices)).unique()
                lo = mins[key].cpu().tolist()
                hi = maxs[key].cpu().tolist()
                for pair, mean, minimum, maximum in zip(range(len(self.pairs)), means, lo, hi, strict=True):
                    rows.append({"benchmark": benchmark, "id": example_id, "method": method, "layer": layer, "pair": pair, "mean_cosine": mean, "min_cosine": minimum, "max_cosine": maximum, "reasoning_tokens": len(positions)})
        if write_selected is not None:
            for token_position in positions:
                for layer in LAYERS:
                    hidden = F.normalize(self.captured[layer][token_position].float(), dim=-1)
                    for method in METHODS:
                        pair_ids = selected_pairs[method, layer]
                        values = (hidden @ self.vectors[method, layer][pair_ids].T).cpu().tolist()
                        write_selected([
                            {"benchmark": benchmark, "id": example_id, "method": method, "layer": layer, "token_index": token_position, "token": token_ids[token_position], "pair": pair, "cosine": value}
                            for pair, value in zip(pair_ids.cpu().tolist(), values, strict=True)
                        ])
        scales = {}
        for key, histogram in histograms.items():
            target = math.ceil(histogram.sum().item() * 0.99)
            scales[f"{key[0]}:{key[1]}"] = max((histogram.cumsum(0) >= target).nonzero()[0].item() / 1023, 1e-6)
        return AnalysisResult(
            len(positions),
            status,
            [self.tokenizer.decode([token_ids[position]], skip_special_tokens=False) for position in positions],
            scales,
            rows,
            [] if write_highlights is not None else highlights,
        )

    def _highlights(self, scores: torch.Tensor, benchmark: str, example_id: str, method: str, layer: int, position: int, token_id: int) -> list[dict[str, Any]]:
        k = min(self.highlights_per_sign, scores.numel())
        positive, positive_ids = torch.topk(scores, k)
        negative, negative_ids = torch.topk(-scores, k)
        token = self.tokenizer.decode([token_id], skip_special_tokens=False)
        rows = []
        for polarity, values, ids in (("positive", positive, positive_ids), ("negative", -negative, negative_ids)):
            for value, pair in zip(values.cpu().tolist(), ids.cpu().tolist(), strict=True):
                rows.append({"benchmark": benchmark, "id": example_id, "method": method, "layer": layer, "token_index": position, "token": token, "pair": pair, "cosine": value, "polarity": polarity})
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
        self.selected_writer: pq.ParquetWriter | None = None
        self.highlight_buffer: list[dict[str, Any]] = []
        self.selected_buffer: list[dict[str, Any]] = []

    def add(self, result: AnalysisResult) -> None:
        self.score_writer = self._write(self.score_writer, self.root / f"concept_scores{self.suffix}.parquet", result.score_rows)
        self.add_highlights(result.highlight_rows)

    def add_highlights(self, rows: list[dict[str, Any]]) -> None:
        self.highlight_buffer.extend(rows)
        if len(self.highlight_buffer) >= 50_000:
            self.highlight_writer = self._write(self.highlight_writer, self.root / f"token_highlights{self.suffix}.parquet", self.highlight_buffer)
            self.highlight_buffer.clear()

    def add_selected(self, rows: list[dict[str, Any]]) -> None:
        self.selected_buffer.extend(rows)
        if len(self.selected_buffer) >= 50_000:
            self.selected_writer = self._write(self.selected_writer, self.root / f"selected_token_activations{self.suffix}.parquet", self.selected_buffer)
            self.selected_buffer.clear()

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
        if self.selected_buffer:
            self.selected_writer = self._write(self.selected_writer, self.root / f"selected_token_activations{self.suffix}.parquet", self.selected_buffer)
            self.selected_buffer.clear()
        for writer in (self.score_writer, self.highlight_writer, self.selected_writer):
            if writer is not None:
                writer.close()
