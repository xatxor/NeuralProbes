"""Serve the concept viewer and compact trace slices on demand."""

from __future__ import annotations

import argparse
import json
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

AVAILABLE_LAYERS = (11, 14, 18, 22, 25)
AVAILABLE_METHODS = ("diff", "concept_centered", "antagonist_centered")
SCOPES = ("reasoning", "full")


class Handler(SimpleHTTPRequestHandler):
    _normalization_cache: dict[str, tuple[int, dict[str, object]]] = {}

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _parameter(self, query: dict[str, list[str]], name: str) -> str:
        values = query.get(name)
        if not values or not values[0]:
            raise ValueError(f"Missing query parameter: {name}")
        return values[0]

    def _trace_root(self, benchmark: str, example_id: str) -> Path:
        if benchmark not in {"aime_2024", "math_500", "gpqa_diamond"}:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in example_id):
            raise ValueError("Invalid example id")
        return Path(self.directory).parent / "traces" / benchmark / example_id

    def _scope(self, query: dict[str, list[str]]) -> str:
        scope = query.get("scope", ["reasoning"])[0]
        if scope not in SCOPES:
            raise ValueError(f"Unknown trace scope: {scope}")
        return scope

    @staticmethod
    def _matrix(
        root: Path,
        meta: dict[str, object],
        method: str,
        layer: int,
        requested_scope: str,
    ) -> tuple[np.ndarray, list[str], str, bool]:
        recorded_methods = tuple(str(value) for value in meta.get("methods", AVAILABLE_METHODS))
        recorded_layers = tuple(int(value) for value in meta.get("layers", AVAILABLE_LAYERS))
        if method not in recorded_methods:
            raise ValueError(f"Method {method} was not recorded for this response")
        if layer not in recorded_layers:
            raise ValueError(f"Layer {layer} was not recorded for this response")

        full_path = root / f"full-{method}-L{layer}.npy"
        if full_path.exists() and isinstance(meta.get("full_tokens"), list):
            matrix = np.load(full_path, mmap_mode="r")
            full_tokens = [str(token) for token in meta["full_tokens"]]
            if requested_scope == "full":
                return matrix, full_tokens, "full", False
            start = max(0, min(int(meta.get("reasoning_start", 0)), matrix.shape[0]))
            end = max(start, min(int(meta.get("reasoning_end", matrix.shape[0])), matrix.shape[0]))
            tokens = meta.get("tokens")
            reasoning_tokens = (
                [str(token) for token in tokens]
                if isinstance(tokens, list)
                else full_tokens[start:end]
            )
            return matrix[start:end], reasoning_tokens, "reasoning", False

        # Backward-compatible fallback for old reasoning-only trace directories.
        reasoning_path = root / f"{method}-L{layer}.npy"
        if not reasoning_path.exists():
            raise FileNotFoundError(
                f"Trace file is missing for method={method}, layer={layer}, id={root.name}"
            )
        tokens = meta.get("tokens", [])
        return (
            np.load(reasoning_path, mmap_mode="r"),
            [str(token) for token in tokens] if isinstance(tokens, list) else [],
            "reasoning",
            requested_scope == "full",
        )

    def _normalization(self) -> dict[str, object]:
        path = Path(self.directory) / "analysis" / "normalization-fp16.npz"
        if not path.exists():
            raise FileNotFoundError("Normalization statistics are missing. Re-run build_concept_report.py.")
        cache_key = str(path)
        modified = path.stat().st_mtime_ns
        cached = self._normalization_cache.get(cache_key)
        if cached is not None and cached[0] == modified:
            return cached[1]
        with np.load(path, allow_pickle=False) as source:
            data: dict[str, object] = {name: source[name] for name in source.files}
        pair_ids = np.asarray(data["pair_ids"], dtype=np.int64)
        data["pair_position"] = {int(pair): index for index, pair in enumerate(pair_ids.tolist())}
        data["method_names"] = tuple(
            str(value) for value in np.asarray(data.get("methods", AVAILABLE_METHODS)).tolist()
        )
        data["layer_values"] = tuple(
            int(value) for value in np.asarray(data.get("layers", AVAILABLE_LAYERS)).tolist()
        )
        self._normalization_cache[cache_key] = (modified, data)
        return data

    def _baseline(
        self,
        method: str,
        layer: int,
        pair_ids: list[int],
        prefix: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self._normalization()
        methods = data["method_names"]
        layers = data["layer_values"]
        if method not in methods:
            raise ValueError(f"Method {method} is missing from normalization statistics")
        if layer not in layers:
            raise ValueError(f"Layer {layer} is missing from normalization statistics")
        positions = data["pair_position"]
        try:
            indices = np.asarray([positions[int(pair)] for pair in pair_ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Pair {error.args[0]} is missing from normalization statistics") from error
        key = methods.index(method), layers.index(layer), indices
        mean = np.asarray(data[f"{prefix}_mean"][key], dtype=np.float32)
        std = np.asarray(data[f"{prefix}_std"][key], dtype=np.float32)
        return mean, std

    def _scope_baseline(
        self, method: str, layer: int, pair_ids: list[int], scope: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if scope == "full":
            data = self._normalization()
            if "full_token_mean" in data:
                mean, std = self._baseline(method, layer, pair_ids, "full_token")
                count = np.asarray(
                    data.get("full_token_count", np.ones_like(data["full_token_mean"]))[
                        data["method_names"].index(method), data["layer_values"].index(layer),
                        [data["pair_position"][int(pair)] for pair in pair_ids],
                    ],
                    dtype=np.int64,
                )
                if np.all(count > 0):
                    return mean, std, "full-response token baseline"
        mean, std = self._baseline(method, layer, pair_ids, "token")
        return mean, std, "reasoning-token baseline"

    def _trace(self, query: dict[str, list[str]]) -> None:
        benchmark = self._parameter(query, "benchmark")
        example_id = self._parameter(query, "id")
        method = self._parameter(query, "method")
        layer = int(self._parameter(query, "layer"))
        pair = int(self._parameter(query, "pair"))
        requested_scope = self._scope(query)
        root = self._trace_root(benchmark, example_id)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        try:
            pair_index = meta["pair_ids"].index(pair)
        except ValueError as error:
            raise ValueError(f"Pair {pair} was not recorded for this response") from error
        matrix, tokens, scope, fallback = self._matrix(
            root, meta, method, layer, requested_scope
        )
        values = np.asarray(matrix[:, pair_index], dtype=np.float32)
        mean, std, baseline_scope = self._scope_baseline(method, layer, [pair], scope)
        z_values = np.full(values.shape, np.nan, dtype=np.float32)
        if std[0] > 1e-6:
            z_values = (values - mean[0]) / std[0]
        self._json(
            {
                "tokens": tokens[: len(values)],
                "values": values.tolist(),
                "z_values": [float(value) if np.isfinite(value) else None for value in z_values],
                "requested_scope": requested_scope,
                "scope": scope,
                "fallback": fallback,
                "baseline": {"mean": float(mean[0]), "std": float(std[0]), "scope": baseline_scope, "dtype": "float16"},
            }
        )

    def _token_concepts(self, query: dict[str, list[str]]) -> None:
        benchmark = self._parameter(query, "benchmark")
        example_id = self._parameter(query, "id")
        method = self._parameter(query, "method")
        layer = int(self._parameter(query, "layer"))
        token_index = int(self._parameter(query, "token_index"))
        requested_scope = self._scope(query)
        root = self._trace_root(benchmark, example_id)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        matrix, tokens, scope, fallback = self._matrix(
            root, meta, method, layer, requested_scope
        )
        if token_index < 0 or token_index >= matrix.shape[0]:
            raise ValueError(f"Token index {token_index} is outside 0..{matrix.shape[0] - 1}")
        values = np.asarray(matrix[token_index], dtype=np.float32)
        pair_ids = [int(pair) for pair in meta["pair_ids"]]
        mean, std, baseline_scope = self._scope_baseline(method, layer, pair_ids, scope)
        z_scores = np.full(values.shape, np.nan, dtype=np.float32)
        valid = std > 1e-6
        z_scores[valid] = (values[valid] - mean[valid]) / std[valid]
        sort_values = np.where(np.isfinite(z_scores), z_scores, -np.inf)
        order = np.argsort(-sort_values, kind="stable")
        self._json(
            {
                "token_index": token_index,
                "token": tokens[token_index],
                "requested_scope": requested_scope,
                "scope": scope,
                "fallback": fallback,
                "normalization": baseline_scope,
                "items": [
                    {
                        "pair": pair_ids[index],
                        "cosine": float(values[index]),
                        "z_score": float(z_scores[index]) if np.isfinite(z_scores[index]) else None,
                        "mean": float(mean[index]),
                        "std": float(std[index]),
                    }
                    for index in order.tolist()
                ],
            }
        )

    def _response_concepts(self, query: dict[str, list[str]]) -> None:
        benchmark = self._parameter(query, "benchmark")
        example_id = self._parameter(query, "id")
        method = self._parameter(query, "method")
        layer = int(self._parameter(query, "layer"))
        requested_scope = self._scope(query)
        root = self._trace_root(benchmark, example_id)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        matrix, tokens, scope, fallback = self._matrix(
            root, meta, method, layer, requested_scope
        )
        values = np.asarray(matrix, dtype=np.float32).mean(axis=0, dtype=np.float32)
        pair_ids = [int(pair) for pair in meta["pair_ids"]]
        mean, std, baseline_scope = self._scope_baseline(method, layer, pair_ids, scope)
        z_scores = np.full(values.shape, np.nan, dtype=np.float32)
        valid = std > 1e-6
        z_scores[valid] = (values[valid] - mean[valid]) / std[valid]
        order = np.argsort(-np.where(np.isfinite(z_scores), z_scores, -np.inf), kind="stable")
        self._json(
            {
                "requested_scope": requested_scope,
                "scope": scope,
                "fallback": fallback,
                "token_count": min(len(tokens), int(matrix.shape[0])),
                "normalization": baseline_scope,
                "items": [
                    {
                        "pair": pair_ids[index],
                        "cosine": float(values[index]),
                        "z_score": float(z_scores[index]) if np.isfinite(z_scores[index]) else None,
                        "mean": float(mean[index]),
                        "std": float(std[index]),
                    }
                    for index in order.tolist()
                ],
            }
        )

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/trace":
                self._trace(query)
                return
            if parsed.path == "/api/token-concepts":
                self._token_concepts(query)
                return
            if parsed.path == "/api/response-concepts":
                self._response_concepts(query)
                return
            super().do_GET()
        except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, status=400)
        except Exception as error:  # keep browser errors inspectable without crashing the server
            self._json({"error": f"Internal viewer error: {error}"}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("01_eval/results"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    directory = args.results.resolve() / "concept_viewer"
    if not (directory / "index.json").exists():
        raise SystemExit(f"Viewer data not found at {directory}. Run build_concept_report.py first.")

    # Keep the generated data files, but always serve the HTML template that belongs
    # to this server version. This prevents a stale results/concept_viewer/index.html
    # from calling API endpoints removed or added by a newer viewer_server.py.
    template = Path(__file__).resolve().with_name("concept_viewer.html")
    if not template.exists():
        raise SystemExit(f"Viewer template not found at {template}")
    shutil.copyfile(template, directory / "index.html")

    handler = lambda *handler_args, **handler_kwargs: Handler(  # noqa: E731
        *handler_args, directory=str(directory), **handler_kwargs
    )
    print(f"Serving concept viewer at http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
