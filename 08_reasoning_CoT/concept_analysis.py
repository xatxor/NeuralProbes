"""Small, streaming concept-vector scorer for Qwen3 evaluation outputs."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
DEFAULT_LAYERS = (18, 22)
DEFAULT_METHODS = ("diff",)
VERSION = 8


def _download(name: str) -> str:
    return hf_hub_download(VECTOR_REPO, name, revision=VECTOR_REVISION)


@dataclass
class AnalysisResult:
    reasoning_token_count: int
    reasoning_status: str
    tokens: list[str]
    score_rows: list[dict[str, Any]]
    analysis_seconds: float = 0.0


class ConceptScorer:
    """Hooks requested residual streams and scores them without saving activations."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        trace_root: Path,
        pair_ids: list[int] | None = None,
        activation_chunk_size: int = 512,
        methods: tuple[str, ...] = DEFAULT_METHODS,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trace_root = trace_root
        self.activation_chunk_size = activation_chunk_size
        if activation_chunk_size < 1:
            raise ValueError("activation_chunk_size must be at least 1")
        self.methods = tuple(dict.fromkeys(methods))
        self.layers = tuple(sorted(set(layers)))
        if not self.methods or any(method not in METHODS for method in self.methods):
            raise ValueError(f"Concept methods must be selected from {METHODS}")
        if not self.layers or any(layer not in LAYERS for layer in self.layers):
            raise ValueError(f"Concept layers must be selected from {LAYERS}")

        self.pairs = pd.read_parquet(_download("pairs.parquet"))
        if len(self.pairs) != 1036:
            raise ValueError(f"Expected 1036 concept pairs, found {len(self.pairs)}")
        self.pair_ids = list(range(len(self.pairs))) if pair_ids is None else sorted(set(pair_ids))
        if not self.pair_ids or min(self.pair_ids) < 0 or max(self.pair_ids) >= len(self.pairs):
            raise ValueError("Concept pair IDs must be between 0 and 1035")

        self.vectors = self._load_vectors()
        self.combined_vectors = {
            layer: torch.cat(
                [self.vectors[method, layer][self.pair_ids] for method in self.methods],
                dim=0,
            )
            for layer in self.layers
        }
        self.captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
        self.handles: list[Any] = []
        self.streams: dict[tuple[str, int], Any] = {}
        self.temp_dir: Path | None = None
        self.captured_tokens = 0

    def _load_vectors(self) -> dict[tuple[str, int], torch.Tensor]:
        tensors: dict[tuple[str, int], torch.Tensor] = {}
        raw: dict[str, torch.Tensor] = {}
        for method in self.methods:
            path = _download(f"{method}.safetensors")
            with safe_open(path, framework="pt") as handle:
                metadata = handle.metadata() or {}
                tensor_name = next(iter(handle.keys()))
                if tuple(handle.get_slice(tensor_name).get_shape()) != (5, 1036, 4096):
                    raise ValueError(f"Unexpected {method} tensor shape")
                manifest = json.loads(metadata.get("manifest", "{}"))
                if manifest.get("layers") != list(LAYERS):
                    raise ValueError(f"Unexpected layers in {method}")
            raw[method] = load_file(path, device=str(self.device))[method].to(dtype=torch.float16)

        if set(METHODS).issubset(raw) and not torch.allclose(
            raw["diff"].float(),
            raw["concept_centered"].float() - raw["antagonist_centered"].float(),
            atol=2e-3,
            rtol=1e-3,
        ):
            raise ValueError("Vector files fail diff = concept_centered - antagonist_centered validation")

        for method, vector in raw.items():
            for layer in self.layers:
                layer_index = LAYERS.index(layer)
                tensors[method, layer] = F.normalize(
                    vector[layer_index].float(), dim=-1
                ).to(dtype=torch.float16)
        return tensors

    def _hook(self, layer: int) -> Callable[..., None]:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            residual = output[0] if isinstance(output, tuple) else output
            # Cached decoding has one input token. Prompt passes are deliberately ignored.
            if residual.ndim == 3 and residual.shape[1] == 1:
                self.captured[layer].append(residual[0, 0].detach())
                if layer == self.layers[-1] and len(self.captured[layer]) >= self.activation_chunk_size:
                    self._flush_captured()

        return capture

    def begin(self, benchmark: str, example_id: str) -> None:
        self.captured = {layer: [] for layer in self.layers}
        self.captured_tokens = 0
        self.temp_dir = self.trace_root / ".tmp" / benchmark / f"{example_id}-{os.getpid()}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.streams = {
            (method, layer): (self.temp_dir / f"{method}-L{layer}.bin").open("wb")
            for method in self.methods
            for layer in self.layers
        }
        blocks = self.model.model.layers
        self.handles = [
            blocks[layer - 1].register_forward_hook(self._hook(layer))
            for layer in self.layers
        ]

    def _flush_captured(self) -> None:
        count = min(len(values) for values in self.captured.values())
        if not count:
            return
        for layer in self.layers:
            hidden = F.normalize(
                torch.stack(self.captured[layer][:count]).float(), dim=-1
            ).to(dtype=torch.float16)
            vectors = self.combined_vectors[layer].to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            projected = hidden @ vectors.T
            # Transfer the complete method block once per layer, then split it on CPU.
            projected_cpu = projected.detach().to(dtype=torch.float16).cpu().numpy()
            for method_index, method in enumerate(self.methods):
                start = method_index * len(self.pair_ids)
                end = start + len(self.pair_ids)
                projected_cpu[:, start:end].tofile(self.streams[method, layer])
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
        self.captured = {layer: [] for layer in self.layers}

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
        start, end, status = thinking_span(self.tokenizer, token_ids, available)
        if start is None:
            self._cleanup_temp()
            return AnalysisResult(0, status, [], [])
        if start == end:
            self._cleanup_temp()
            return AnalysisResult(0, "empty_thinking", [], [])

        n_pairs = len(self.pair_ids)
        sums = {
            (method, layer): np.zeros(n_pairs, dtype=np.float32)
            for method in self.methods
            for layer in self.layers
        }
        trace_dir = self.trace_root / benchmark / str(example_id)
        trace_dir.mkdir(parents=True, exist_ok=True)

        for method in self.methods:
            for layer in self.layers:
                source = np.memmap(
                    self.temp_dir / f"{method}-L{layer}.bin",
                    dtype=np.float16,
                    mode="r",
                    shape=(self.captured_tokens, n_pairs),
                )

                # Store the complete generated continuation once. Reasoning views are
                # sliced from this file using reasoning_start/reasoning_end in meta.json.
                target_path = trace_dir / f"full-{method}-L{layer}.tmp.npy"
                target = np.lib.format.open_memmap(
                    target_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(available, n_pairs),
                )
                for offset in range(0, available, self.activation_chunk_size):
                    values = source[offset : min(available, offset + self.activation_chunk_size)]
                    target[offset : offset + len(values)] = values
                del target
                target_path.replace(trace_dir / f"full-{method}-L{layer}.npy")

                # Preserve the original reasoning-span mean-cosine calculation,
                # without writing a second reasoning-only trace.
                key = method, layer
                for offset in range(0, end - start, self.activation_chunk_size):
                    values = source[
                        start + offset : min(end, start + offset + self.activation_chunk_size)
                    ]
                    sums[key] += values.astype(np.float32).sum(axis=0)
                del source

        rows: list[dict[str, Any]] = []
        for method in self.methods:
            for layer in self.layers:
                means = (sums[method, layer] / (end - start)).tolist()
                for pair, mean in zip(self.pair_ids, means, strict=True):
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "id": example_id,
                            "method": method,
                            "layer": layer,
                            "pair": pair,
                            "mean_cosine": mean,
                            "reasoning_tokens": end - start,
                        }
                    )

        # batch_decode on singleton sequences is equivalent to decoding each token
        # independently, while avoiding repeated tokenizer setup overhead.
        full_tokens = self.tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=False,
        )
        reasoning_tokens = full_tokens[start:end]
        (trace_dir / "meta.tmp.json").write_text(
            json.dumps(
                {
                    "pair_ids": self.pair_ids,
                    "methods": list(self.methods),
                    "layers": list(self.layers),
                    "tokens": reasoning_tokens,
                    "full_tokens": full_tokens,
                    "reasoning_start": start,
                    "reasoning_end": end,
                    "dtype": "float16",
                },
                ensure_ascii=False,
            )
        )
        (trace_dir / "meta.tmp.json").replace(trace_dir / "meta.json")
        self._cleanup_temp()
        return AnalysisResult(
            end - start,
            status,
            reasoning_tokens,
            rows,
            time.perf_counter() - analysis_started,
        )


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
    """Append score rows without discarding compatible rows from an earlier resume."""

    SCORE_COLUMNS = (
        "benchmark",
        "id",
        "method",
        "layer",
        "pair",
        "mean_cosine",
        "reasoning_tokens",
    )
    SCORE_KEY = ("benchmark", "id", "method", "layer", "pair")

    def __init__(self, root: Path, suffix: str) -> None:
        self.root = root
        self.suffix = suffix
        self.score_path = self.root / f"concept_scores{self.suffix}.parquet"
        self.pending_path = self.root / (
            f".concept_scores{self.suffix}.{os.getpid()}.pending.parquet"
        )
        self.pending_path.unlink(missing_ok=True)
        self.score_writer: pq.ParquetWriter | None = None

    def add(self, result: AnalysisResult) -> None:
        self.score_writer = self._write(
            self.score_writer,
            self.pending_path,
            result.score_rows,
        )

    @staticmethod
    def _write(
        writer: pq.ParquetWriter | None,
        path: Path,
        rows: list[dict[str, Any]],
    ) -> pq.ParquetWriter | None:
        if not rows:
            return writer
        path.parent.mkdir(exist_ok=True)
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        return writer

    @classmethod
    def _read_scores(cls, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        missing = [column for column in cls.SCORE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Score file {path} is missing columns: {missing}")
        return frame.loc[:, cls.SCORE_COLUMNS]

    def close(self) -> None:
        if self.score_writer is not None:
            self.score_writer.close()
            self.score_writer = None
        if not self.pending_path.exists():
            return

        frames = []
        if self.score_path.exists():
            frames.append(self._read_scores(self.score_path))
        frames.append(self._read_scores(self.pending_path))
        merged = pd.concat(frames, ignore_index=True)
        merged["id"] = merged["id"].astype(str)
        merged = merged.drop_duplicates(list(self.SCORE_KEY), keep="last")
        merged = merged.sort_values(list(self.SCORE_KEY), kind="stable")

        replacement = self.score_path.with_suffix(self.score_path.suffix + ".tmp")
        merged.to_parquet(replacement, index=False, compression="zstd")
        replacement.replace(self.score_path)
        self.pending_path.unlink(missing_ok=True)
