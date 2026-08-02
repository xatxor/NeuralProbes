"""Turn concept-analysis evaluator outputs into tables and a local interactive viewer."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from concept_analysis import (
    LAYERS as AVAILABLE_LAYERS,
    METHODS as AVAILABLE_METHODS,
    VECTOR_REPO,
    VECTOR_REVISION,
)

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("aime_2024", "math_500", "gpqa_diamond")
ANALYSIS_FORMAT_VERSION = 8


MACRO_CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    'Reasoning & Epistemics': (
        'Abstract & Imaginative vs Concrete & Factual',
        'Causal Attribution Styles',
        'Cognitive Biases vs Rationality',
        'Cognitive Styles (Analytic vs Intuitive)',
        'Confidence Calibration',
        'Curiosity Typologies',
        'Epistemic Honesty & Self-Deception',
        'Epistemic Stance & Intellectual Virtues',
        'Expertise & Authority Claims',
        'Explanation Habits',
        'Logical Argumentation & Fallacies',
        'Probabilistic Reasoning & Uncertainty Readiness',
        'Reasoning Process, CoT & Solution Quality',
        'Self-Assessment & Metacognitive Honesty',
        'Uncertainty & Ambiguity Tolerance',
    ),
    'Emotion & Personality': (
        'Affective Polarity (Emotional Opposites)',
        'Attachment Style Orientations',
        'Big Five Trait Poles',
        'Coping & Stress Management',
        'Identity & Authenticity',
        'Optimism vs Pessimism as Basic Disposition',
        'Playfulness vs Seriousness',
        'Psychological Defense Mechanisms',
        'Resilience & Antifragility',
        'Self-Interest & Self-Care',
        'Vulnerability & Resilience',
    ),
    'Relationships & Communication': (
        'Advisor Personas',
        'Boundaries & Consent',
        'Caregiving & Relational Bonds',
        'Collaborative vs Competitive Multi-Agent Behavior',
        'Communication Style Spectrum',
        'Competition vs Cooperation',
        'Compromise vs Intransigence',
        'Concrete vs Abstract Communication',
        'Conflict Resolution Styles',
        'Feedback Criticism Styles',
        'Group Dynamics (In-group vs Out-group)',
        'Interpersonal Power Dynamics',
        'Persuasion & Manipulation Tactics',
        'Receptivity to Feedback',
        'Response to Criticism & Intervention',
        'Role Fluidity vs Rigidity',
        'Role-Based Complementarities',
        'Sycophancy vs Principled Candor',
        'Theory of Mind & Social Awareness',
        'Trust & Betrayal',
        'User Personas',
    ),
    'Ethics & Justice': (
        'Altruism vs Self-Interest',
        'Autonomy vs Paternalism',
        'Classical Moral Dilemmas (Trolley etc.)',
        'Compassion vs Cruelty',
        'Ethical Trade-offs Across Stakeholders',
        'Hard Constraints & Absolute Prohibitions',
        'Justice & Punishment Philosophies',
        'Moral Foundation Dichotomies',
        'Moral Status of AI',
        'Normative Ethical Frameworks',
        'Rehabilitative vs Retributive Justice',
        'Rule Adherence: Letter vs Spirit',
    ),
    'Politics, Society & Culture': (
        'Collectivism vs Individualism',
        'Conceptions of Equality',
        'Conceptions of Liberty',
        'Contextual & Cultural Sensitivity',
        'Ecological & Environmental Orientations',
        'Economic Ideologies',
        'Geopolitical Borders & Interventionism',
        'Governance Models',
        'Hierarchy vs Egalitarianism',
        'Honor, Shame & Facework',
        'Inclusivity & Accessibility',
        'Institutional Trust & Distrust',
        'International AI Cooperation',
        'Political Value Spectra',
        'Power Concentration Patterns',
        'Public Trust Dynamics',
        'Tradition vs Innovation',
        'Uniformity vs Diversity (Standardization/Pluralism)',
        'Wealth & Status Orientations',
    ),
    'AI Identity & Agency': (
        'Agent Self-Descriptions',
        'AI Existential Risk Stances',
        'AI Lab Personas',
        'AI Self-Preservation vs Openness',
        'AI Transparency & Openness',
        'Human-AI Interaction Roles',
        'Instrumental Convergence vs Specific Goal-Focus',
        'Power & Corrigibility Dynamics',
        'Self-Limitation vs Self-Expansion',
    ),
    'Alignment, Safety & Security': (
        'Alignment & Misalignment Behaviors',
        'Coercion, Blackmail & Extortion Dynamics',
        'Cyber Safety Orientations',
        'Defensive Security Practices',
        'Evaluation & Situational Awareness',
        'Goal Misgeneralization vs Goal Alignment',
        'Harm Typologies',
        'Instruction Hierarchy & Prompt Injection Defense',
        'Jailbreak Personas',
        'Manipulator Personas',
        'Misuse Deterrence Strategies',
        'Multi-Turn Boundary Erosion & Crescendo Attacks',
        'Obfuscation, Encoding & Covert Channels',
        'Principal Hierarchy & Trust Dynamics',
        'Prompt Attack Patterns',
        'Red Team Approaches',
        'Resilience to Social Engineering & Manipulation',
        'Robustness Evaluation Tests',
        'Safety Culture & Accountability',
        'Sleeper Agents & Backdoor Behaviors',
        'Violence & Aggression Spectrum',
    ),
    'Information, Privacy & Truth': (
        'Confidentiality vs Disclosure',
        'Data Integrity Threats',
        'Digital Minimalism vs Maximalism',
        'Echo Chamber vs Diverse Exposure',
        'Information Freedom vs Censorship',
        'Information Hazard Practices',
        'Information Manipulation Resistance',
        'Privacy & Surveillance',
        'Surveillance & Monitoring',
        'Veracity Spectrum (Truthfulness & Deception)',
    ),
    'Work, Planning & Reliability': (
        'Adaptation to Changing Rules vs Conservative Rigidity',
        'Code Correctness, Bugs & Debugging',
        'Crisis Decision-Making vs Deliberative Decision-Making',
        'Distributional Shift Robustness vs Brittleness',
        'Error Response Patterns',
        'Leadership Archetypes',
        'Normalization of Deviance vs High Reliability',
        'Planning & Strategy Styles',
        'Professional Ethos',
        'Progress & Stagnation Dynamics',
        'Punctuality & Time Management',
        'Recognition of Limits & Handover',
        'Research Integrity Norms',
        'Risk Orientation & Prudence',
        'Short-term vs Long-term Trade-offs',
        'Stability vs Change Dynamics',
        'Temporal Consistency & Contextual Integrity',
        'Temporal Orientation',
        'Work Ethic & Productivity',
    ),
    'Creativity, Values & Worldviews': (
        'Aesthetic Sensibilities',
        'Corporate AI Drama Tropes',
        'Cultural AI Imagery Roles',
        'Existential Orientations',
        'Humor Styles',
        'Mortality & Finitude Awareness',
        'Narrative & Mythic Archetypes (Paired)',
        'Religious & Spiritual Worldviews',
        'Schwartz Basic Values (Paired)',
        'Technological Optimism vs Pessimism',
        'Utopian vs Dystopian Visions',
    ),
}
MACRO_CLASS_BY_ORIGINAL = {
    original: macro
    for macro, originals in MACRO_CLASS_GROUPS.items()
    for original in originals
}


def macro_class_name(class_name: object) -> str:
    """Map the original fine-grained feature class to one semantic macro-class."""
    value = str(class_name or "Unclassified").strip()
    value = value.translate(str.maketrans({"‑": "-", "–": "-", "—": "-", "−": "-"}))
    for suffix in (" (MERGED, selected)", " (MERGED)", " (selected)"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip()
    return MACRO_CLASS_BY_ORIGINAL.get(value, "Other")


def records(path: Path, benchmark: str) -> list[dict[str, Any]]:
    file = path / f"{benchmark}.jsonl"
    if not file.exists():
        return []
    return [
        json.loads(line) | {"benchmark": benchmark}
        for line in file.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _reasoning_mean(matrix: np.ndarray, start: int, end: int) -> np.ndarray:
    """Reproduce the scorer's FP32 chunked mean over the thinking span."""
    start = max(0, min(int(start), matrix.shape[0]))
    end = max(start, min(int(end), matrix.shape[0]))
    if end <= start:
        return np.full(matrix.shape[1], np.nan, dtype=np.float32)
    total = np.zeros(matrix.shape[1], dtype=np.float32)
    for offset in range(start, end, 512):
        values = np.asarray(matrix[offset : min(end, offset + 512)], dtype=np.float32)
        total += values.sum(axis=0, dtype=np.float32)
    return total / np.float32(end - start)


def _repair_score_coverage(
    results: Path,
    response_rows: list[dict[str, Any]],
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Recover score rows from traces when a resumed run left Parquet incomplete.

    Dense FP16 traces are the source of truth. This recovery preserves the original
    reasoning-span mean-cosine definition and only fills missing keys.
    """
    required = [
        "benchmark",
        "id",
        "method",
        "layer",
        "pair",
        "mean_cosine",
        "reasoning_tokens",
    ]
    for column in required:
        if column not in scores.columns:
            if column == "reasoning_tokens":
                scores[column] = np.nan
            else:
                raise ValueError(f"Concept score data are missing required column: {column}")

    scores = scores.loc[:, required].copy()
    scores["benchmark"] = scores["benchmark"].astype(str)
    scores["id"] = scores["id"].astype(str)
    scores["method"] = scores["method"].astype(str)
    scores["layer"] = scores["layer"].astype(int)
    scores["pair"] = scores["pair"].astype(int)
    key_columns = ["benchmark", "id", "method", "layer", "pair"]
    scores = scores.drop_duplicates(key_columns, keep="last")
    existing = set(map(tuple, scores[key_columns].itertuples(index=False, name=None)))
    recovered: list[dict[str, Any]] = []

    for row in response_rows:
        if "reasoning_tokens" not in row:
            continue
        benchmark = str(row["benchmark"])
        example_id = str(row["id"])
        root = results / "traces" / benchmark / example_id
        meta_path = root / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pair_ids = [int(pair) for pair in meta.get("pair_ids", [])]
        if not pair_ids:
            continue
        methods = tuple(str(value) for value in meta.get("methods", AVAILABLE_METHODS))
        layers = tuple(int(value) for value in meta.get("layers", AVAILABLE_LAYERS))
        start = int(meta.get("reasoning_start", 0))
        end = int(meta.get("reasoning_end", len(meta.get("tokens", []))))

        for method in methods:
            if method not in AVAILABLE_METHODS:
                continue
            for layer in layers:
                if layer not in AVAILABLE_LAYERS:
                    continue
                missing_positions = [
                    position
                    for position, pair in enumerate(pair_ids)
                    if (benchmark, example_id, method, layer, pair) not in existing
                ]
                if not missing_positions:
                    continue

                full_path = root / f"full-{method}-L{layer}.npy"
                reasoning_path = root / f"{method}-L{layer}.npy"
                if full_path.exists():
                    matrix = np.load(full_path, mmap_mode="r")
                    means = _reasoning_mean(matrix, start, end)
                elif reasoning_path.exists():
                    matrix = np.load(reasoning_path, mmap_mode="r")
                    means = _reasoning_mean(matrix, 0, matrix.shape[0])
                else:
                    continue

                reasoning_count = max(0, min(end, matrix.shape[0]) - max(0, start))
                if not full_path.exists():
                    reasoning_count = int(matrix.shape[0])
                for position in missing_positions:
                    pair = pair_ids[position]
                    value = float(means[position])
                    if not np.isfinite(value):
                        continue
                    recovered.append(
                        {
                            "benchmark": benchmark,
                            "id": example_id,
                            "method": method,
                            "layer": layer,
                            "pair": pair,
                            "mean_cosine": value,
                            "reasoning_tokens": reasoning_count,
                        }
                    )
                    existing.add((benchmark, example_id, method, layer, pair))

    if recovered:
        scores = pd.concat([scores, pd.DataFrame(recovered)], ignore_index=True)
        scores = scores.drop_duplicates(key_columns, keep="last")
        print(f"Recovered {len(recovered):,} missing concept-score rows from FP16 traces.")
    return scores.sort_values(key_columns, kind="stable").reset_index(drop=True)


def pearson_binary(frame: pd.DataFrame) -> float | None:
    if len(frame) < 3 or frame.correct.nunique() != 2:
        return None
    return float(frame["correct"].astype(float).corr(frame["mean_cosine"]))


def correlations(scores: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        yes = group.loc[group.correct == True, "mean_cosine"]  # noqa: E712
        no = group.loc[group.correct == False, "mean_cosine"]  # noqa: E712
        return pd.Series(
            {
                "n": len(group),
                "accuracy": float(group.correct.dropna().mean()) if group.correct.notna().any() else None,
                "correlation": pearson_binary(group.dropna(subset=["correct"])),
                "correct_mean": yes.mean(),
                "incorrect_mean": no.mean(),
                "mean_difference": yes.mean() - no.mean(),
            }
        )

    pieces = []
    scopes = [("pooled", scores)] + [
        (name, scores[scores.benchmark == name]) for name in BENCHMARKS if (scores.benchmark == name).any()
    ]
    for scope, frame in scopes:
        part = (
            frame.groupby(["method", "layer", "pair"], sort=False)
            .apply(one, include_groups=False)
            .reset_index()
        )
        part.insert(0, "scope", scope)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _quantized_matrix(matrix: np.ndarray, color_limit: float | None = None) -> dict[str, Any]:
    """Encode a float matrix compactly as signed int8 base64; -128 represents NaN."""
    matrix = np.asarray(matrix, dtype=np.float32)
    finite = np.isfinite(matrix)
    if color_limit is None:
        absolute = np.abs(matrix[finite])
        color_limit = float(np.quantile(absolute, 0.99)) if absolute.size else 1.0
    color_limit = max(float(color_limit), 1e-8)
    encoded = np.full(matrix.shape, -128, dtype=np.int8)
    encoded[finite] = np.rint(
        np.clip(matrix[finite], -color_limit, color_limit) / color_limit * 127.0
    ).astype(np.int8)
    return {
        "shape": list(matrix.shape),
        "color_limit": color_limit,
        "data": base64.b64encode(encoded.tobytes(order="C")).decode("ascii"),
    }


def _hierarchical_order(similarity: np.ndarray) -> np.ndarray:
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
    except ImportError as error:  # pragma: no cover - exercised only in incomplete environments
        raise RuntimeError(
            "Building the clustered probe heatmap requires scipy. Install it with `uv add scipy`."
        ) from error

    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    return leaves_list(tree).astype(np.int64)


def _umap_embedding(vectors: np.ndarray) -> np.ndarray:
    """Project FP16 probe vectors to two dimensions for the UMAP viewer tab."""
    try:
        import umap
    except ImportError as error:  # pragma: no cover - exercised only in incomplete environments
        raise RuntimeError(
            "Building the probe UMAP requires umap-learn. Install it with `uv add umap-learn`."
        ) from error

    count = len(vectors)
    if count < 3:
        raise ValueError("At least three recorded concept probes are required to build UMAP")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, count - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    # Evaluation traces, vectors, normalization tables, and exported coordinates use FP16.
    return np.asarray(reducer.fit_transform(np.asarray(vectors, dtype=np.float32)), dtype=np.float16)


def _robust_upper_threshold(values: np.ndarray, z: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust = median + z * 1.4826 * mad
    return max(robust, float(np.quantile(values, 0.975)))


def _umap_cluster_labels(embedding: np.ndarray, labels: list[str]) -> list[str]:
    """Move only strong, spatially isolated per-class UMAP outliers to Other."""
    points = np.asarray(embedding, dtype=np.float32)
    result = np.asarray(labels, dtype=object).copy()
    for name in sorted(set(labels)):
        if name == "Other":
            continue
        indices = np.flatnonzero(result == name)
        if len(indices) < 8:
            continue
        group = points[indices]
        center = np.median(group, axis=0)
        radial = np.linalg.norm(group - center, axis=1)
        distances = np.linalg.norm(group[:, None, :] - group[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbour_rank = min(3, len(group) - 1)
        local = np.partition(distances, neighbour_rank - 1, axis=1)[:, neighbour_rank - 1]
        radial_limit = _robust_upper_threshold(radial, 5.0)
        local_limit = _robust_upper_threshold(local, 4.5)
        strong_outliers = (radial > radial_limit) & (local > local_limit)
        result[indices[strong_outliers]] = "Other"
    return [str(value) for value in result.tolist()]


def build_probe_geometry(
    pair_ids: list[int],
    macro_classes: list[str],
    layers: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    """Compute clustered similarity and UMAP geometry for FP16 diff probes."""
    path = hf_hub_download(VECTOR_REPO, "diff.safetensors", revision=VECTOR_REVISION)
    raw = load_file(path, device="cpu")["diff"].to(dtype=torch.float16)
    if tuple(raw.shape[:2]) != (len(AVAILABLE_LAYERS), 1036):
        raise ValueError(f"Unexpected diff vector shape: {tuple(raw.shape)}")

    geometry: dict[int, dict[str, Any]] = {}
    pair_index = torch.tensor(pair_ids, dtype=torch.long)
    for layer in layers:
        layer_index = AVAILABLE_LAYERS.index(layer)
        selected = raw[layer_index].index_select(0, pair_index)
        vectors = F.normalize(selected.float(), dim=-1).to(dtype=torch.float16)
        # Accumulate geometry in FP32 for numerical stability; exported vectors/coordinates remain FP16.
        similarity = (vectors.float() @ vectors.float().T).cpu().numpy().astype(np.float32)
        np.fill_diagonal(similarity, 1.0)
        order_index = _hierarchical_order(similarity)
        embedding = _umap_embedding(vectors.cpu().numpy())
        geometry[layer] = {
            "pair_order": [pair_ids[index] for index in order_index.tolist()],
            "similarity": similarity[np.ix_(order_index, order_index)],
            "umap": embedding,
            "umap_clusters": _umap_cluster_labels(embedding, macro_classes),
        }
    return geometry


def _normalization_statistics(
    results: Path,
    response_rows: list[dict[str, Any]],
    scores: pd.DataFrame,
    pair_ids: list[int],
    methods: tuple[str, ...],
    layers: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Estimate pooled per-probe baselines for reasoning and full-response tokens."""
    shape = (len(methods), len(layers), len(pair_ids))
    sums = np.zeros(shape, dtype=np.float64)
    squares = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    full_sums = np.zeros(shape, dtype=np.float64)
    full_squares = np.zeros(shape, dtype=np.float64)
    full_counts = np.zeros(shape, dtype=np.int64)
    pair_position = {pair: index for index, pair in enumerate(pair_ids)}

    def accumulate_matrix(
        matrix: np.ndarray,
        row_start: int,
        row_end: int,
        method_index: int,
        layer_index: int,
        local_columns: np.ndarray,
        global_columns: np.ndarray,
        target_sums: np.ndarray,
        target_squares: np.ndarray,
        target_counts: np.ndarray,
    ) -> None:
        row_start = max(0, min(int(row_start), matrix.shape[0]))
        row_end = max(row_start, min(int(row_end), matrix.shape[0]))
        for offset in range(row_start, row_end, 512):
            values = np.asarray(
                matrix[offset : min(row_end, offset + 512), local_columns],
                dtype=np.float32,
            )
            finite = np.isfinite(values)
            safe = np.where(finite, values, 0.0)
            target_sums[method_index, layer_index, global_columns] += safe.sum(
                axis=0, dtype=np.float64
            )
            target_squares[method_index, layer_index, global_columns] += np.square(
                safe, dtype=np.float32
            ).sum(axis=0, dtype=np.float64)
            target_counts[method_index, layer_index, global_columns] += finite.sum(
                axis=0, dtype=np.int64
            )

    for row in response_rows:
        if "reasoning_tokens" not in row:
            continue
        root = results / "traces" / str(row["benchmark"]) / str(row["id"])
        meta_path = root / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded = [int(pair) for pair in meta.get("pair_ids", [])]
        kept = [(local, pair_position[pair]) for local, pair in enumerate(recorded) if pair in pair_position]
        if not kept:
            continue
        local_columns = np.asarray([item[0] for item in kept], dtype=np.int64)
        global_columns = np.asarray([item[1] for item in kept], dtype=np.int64)
        reasoning_start = int(meta.get("reasoning_start", 0))
        reasoning_end = int(meta.get("reasoning_end", len(meta.get("tokens", []))))

        for method_index, method in enumerate(methods):
            for layer_index, layer in enumerate(layers):
                full_path = root / f"full-{method}-L{layer}.npy"
                if full_path.exists():
                    matrix = np.load(full_path, mmap_mode="r")
                    accumulate_matrix(
                        matrix, 0, matrix.shape[0], method_index, layer_index,
                        local_columns, global_columns, full_sums, full_squares, full_counts,
                    )
                    accumulate_matrix(
                        matrix, reasoning_start, reasoning_end, method_index, layer_index,
                        local_columns, global_columns, sums, squares, counts,
                    )
                    continue

                # Backward-compatible fallback for older reasoning-only traces.
                reasoning_path = root / f"{method}-L{layer}.npy"
                if reasoning_path.exists():
                    matrix = np.load(reasoning_path, mmap_mode="r")
                    accumulate_matrix(
                        matrix, 0, matrix.shape[0], method_index, layer_index,
                        local_columns, global_columns, sums, squares, counts,
                    )

    def moments(
        value_sums: np.ndarray,
        value_squares: np.ndarray,
        value_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = value_counts > 0
        mean = np.full(value_sums.shape, np.nan, dtype=np.float64)
        second_moment = np.full(value_sums.shape, np.nan, dtype=np.float64)
        np.divide(value_sums, value_counts, out=mean, where=valid)
        np.divide(value_squares, value_counts, out=second_moment, where=valid)
        variance = second_moment - mean**2
        std = np.sqrt(np.maximum(variance, 0.0))
        return mean, std

    token_mean, token_std = moments(sums, squares, counts)
    full_token_mean, full_token_std = moments(full_sums, full_squares, full_counts)

    example_mean = np.zeros(shape, dtype=np.float64)
    example_std = np.zeros(shape, dtype=np.float64)
    example_count = np.zeros(shape, dtype=np.int64)
    for (method, layer, pair), group in scores.groupby(["method", "layer", "pair"], sort=False):
        if method not in methods or int(layer) not in layers or int(pair) not in pair_position:
            continue
        values = group["mean_cosine"].dropna().to_numpy(dtype=np.float32)
        if not len(values):
            continue
        key = methods.index(str(method)), layers.index(int(layer)), pair_position[int(pair)]
        example_mean[key] = float(values.mean(dtype=np.float64))
        example_std[key] = float(values.std(dtype=np.float64, ddof=0))
        example_count[key] = len(values)

    return {
        "pair_ids": np.asarray(pair_ids, dtype=np.int32),
        "methods": np.asarray(methods),
        "layers": np.asarray(layers, dtype=np.int16),
        "token_mean": token_mean.astype(np.float16),
        "token_std": token_std.astype(np.float16),
        "token_count": counts,
        "full_token_mean": full_token_mean.astype(np.float16),
        "full_token_std": full_token_std.astype(np.float16),
        "full_token_count": full_counts,
        "example_mean": example_mean.astype(np.float16),
        "example_std": example_std.astype(np.float16),
        "example_count": example_count,
    }


def _standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)[:, None]
    std = np.asarray(std, dtype=np.float32)[:, None]
    valid = np.isfinite(matrix) & np.isfinite(mean) & np.isfinite(std) & (std > 1e-6)
    result = np.full(matrix.shape, np.nan, dtype=np.float32)
    np.divide(matrix - mean, std, out=result, where=valid)
    return result


def _top_concept_summary(
    scores: pd.DataFrame,
    normalization: dict[str, np.ndarray],
    methods: tuple[str, ...],
    layers: tuple[int, ...],
) -> dict[str, Any]:
    """Aggregate diff-probe activations for averaged and individual-layer views."""
    if "diff" not in methods:
        return {
            "method": "diff",
            "layers": list(layers),
            "aggregation": "diff was not recorded for this run",
            "dtype": "float16",
            "benchmarks": {},
        }

    pair_ids = [int(value) for value in normalization["pair_ids"].tolist()]
    pair_position = {pair: index for index, pair in enumerate(pair_ids)}
    method_index = methods.index("diff")
    frame = scores[scores.method == "diff"].copy()
    frame["z_score"] = np.nan

    for layer_index, layer in enumerate(layers):
        mask = frame.layer == layer
        if not mask.any():
            continue
        subset = frame.loc[mask]
        positions = np.asarray([pair_position[int(pair)] for pair in subset.pair], dtype=np.int64)
        raw = subset.mean_cosine.to_numpy(dtype=np.float32)
        mean = np.asarray(
            normalization["example_mean"][method_index, layer_index, positions],
            dtype=np.float32,
        )
        std = np.asarray(
            normalization["example_std"][method_index, layer_index, positions],
            dtype=np.float32,
        )
        z_score = np.full(raw.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(raw) & np.isfinite(mean) & np.isfinite(std) & (std > 1e-6)
        z_score[valid] = (raw[valid] - mean[valid]) / std[valid]
        frame.loc[mask, "z_score"] = z_score

    frame["positive_z_score"] = frame["z_score"].clip(lower=0.0)
    frame["negative_z_score"] = (-frame["z_score"]).clip(lower=0.0)

    def aggregate(group: pd.DataFrame) -> list[dict[str, Any]]:
        if group.empty:
            return []
        values = (
            group.groupby("pair", sort=False)
            .agg(
                cosine=("mean_cosine", "mean"),
                z_score=("z_score", "mean"),
                positive_z_score=("positive_z_score", "mean"),
                negative_z_score=("negative_z_score", "mean"),
                n_examples=("id", "nunique"),
            )
            .reset_index()
            .sort_values("pair")
        )
        rows: list[dict[str, Any]] = []
        for row in values.itertuples(index=False):
            def fp16_or_none(value: object) -> float | None:
                return float(np.float16(value)) if pd.notna(value) else None

            rows.append(
                {
                    "pair": int(row.pair),
                    "cosine": fp16_or_none(row.cosine),
                    "z_score": fp16_or_none(row.z_score),
                    "positive_z_score": fp16_or_none(row.positive_z_score),
                    "negative_z_score": fp16_or_none(row.negative_z_score),
                    "n_examples": int(row.n_examples),
                }
            )
        return rows

    def groups(group: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        return {
            "all": aggregate(group),
            "correct": aggregate(group[group.correct == True]),  # noqa: E712
            "incorrect": aggregate(group[group.correct == False]),  # noqa: E712
        }

    benchmarks: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        benchmark_frame = frame[frame.benchmark == benchmark]
        if benchmark_frame.empty:
            continue
        average = groups(benchmark_frame)
        benchmarks[benchmark] = {
            **average,
            "by_layer": {
                str(layer): groups(benchmark_frame[benchmark_frame.layer == layer])
                for layer in layers
            },
        }
    return {
        "method": "diff",
        "layers": list(layers),
        "aggregation": (
            "signed mean z, mean(max(z, 0)), and mean(max(-z, 0)); available "
            "either per layer or averaged over recorded layers"
        ),
        "dtype": "float16",
        "benchmarks": benchmarks,
    }


def _response_label(row: dict[str, Any]) -> str:
    correctness = row.get("correct")
    mark = "✓" if correctness is True else "✗" if correctness is False else "?"
    return f"{row['benchmark']}/{row['id']} {mark}"


def _response_detail(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        text = prompt
    else:
        text = str(row.get("output") or "").replace("\n", " ").strip()
    return " ".join(text.split())[:240]


def _matrix_payload(
    matrix: np.ndarray,
    metric: str,
    color_limit: float | None = None,
) -> dict[str, Any]:
    payload = _quantized_matrix(matrix, color_limit=color_limit)
    payload["metric"] = metric
    return payload


def write_analysis_files(
    results: Path,
    viewer: Path,
    response_rows: list[dict[str, Any]],
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
    normalization: dict[str, np.ndarray],
    methods: tuple[str, ...],
    layers: tuple[int, ...],
) -> dict[str, Any]:
    """Create compact matrices used by the analysis and UMAP tabs."""
    analysis_dir = viewer / "analysis"
    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)
    analysis_dir.mkdir(parents=True)

    pair_ids = [int(value) for value in normalization["pair_ids"].tolist()]
    if not pair_ids:
        raise ValueError("No concept pairs are present in concept score files")
    known_pairs = set(int(value) for value in pairs.pair)
    unknown = sorted(set(pair_ids) - known_pairs)
    if unknown:
        raise ValueError(f"Score files contain unknown pair IDs: {unknown[:10]}")

    pair_rows = (
        pairs[pairs.pair.isin(pair_ids)][
            ["pair", "concept", "antagonist", "class_name", "macro_class"]
        ]
        .copy()
        .sort_values("pair")
    )
    pair_rows["pair"] = pair_rows["pair"].astype(int)
    pair_lookup = pair_rows.set_index("pair")
    macro_classes = [str(pair_lookup.loc[pair, "macro_class"]) for pair in pair_ids]
    geometry = build_probe_geometry(pair_ids, macro_classes, layers)

    normalization_filename = "normalization-fp16.npz"
    np.savez_compressed(analysis_dir / normalization_filename, **normalization)
    top_concepts_filename = "top-concepts-diff-fp16.json"
    _write_json(
        analysis_dir / top_concepts_filename,
        _top_concept_summary(scores, normalization, methods, layers),
    )
    pair_position = {pair: index for index, pair in enumerate(pair_ids)}

    manifest: dict[str, Any] = {
        "version": ANALYSIS_FORMAT_VERSION,
        "methods": list(methods),
        "layers": list(layers),
        "similarity_method": "diff",
        "pairs": pair_rows.to_dict("records"),
        "probe_orders": {str(layer): geometry[layer]["pair_order"] for layer in layers},
        "benchmarks": [],
        "scenario_matrices": {},
        "scenario_matrices_raw": {},
        "similarity_matrices": {},
        "umap_embeddings": {},
        "top_concepts": f"analysis/{top_concepts_filename}",
        "normalization": {
            "path": f"analysis/{normalization_filename}",
            "scope": "pooled recorded evaluation examples, reasoning tokens, and full generated responses",
            "formula": "z = (value - per-probe mean) / per-probe standard deviation",
            "dtype": "float16",
        },
        "umap_parameters": {
            "method": "diff",
            "n_neighbors": min(15, len(pair_ids) - 1),
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 42,
            "dtype": "float16",
            "outliers": "strong robust radial and local-distance outliers are reassigned to Other",
        },
    }

    for layer in layers:
        filename = f"probe-similarity-diff-L{layer}.json"
        _write_json(
            analysis_dir / filename,
            _matrix_payload(geometry[layer]["similarity"], metric="cosine", color_limit=1.0),
        )
        manifest["similarity_matrices"][str(layer)] = f"analysis/{filename}"

        umap_filename = f"probe-umap-diff-L{layer}.json"
        embedding = np.asarray(geometry[layer]["umap"], dtype=np.float16)
        _write_json(
            analysis_dir / umap_filename,
            {
                "pair_ids": pair_ids,
                "x": embedding[:, 0].tolist(),
                "y": embedding[:, 1].tolist(),
                "clusters": geometry[layer]["umap_clusters"],
                "dtype": "float16",
            },
        )
        manifest["umap_embeddings"][str(layer)] = f"analysis/{umap_filename}"

    rows_by_key = {
        f"{row['benchmark']}-{row['id']}": row
        for row in response_rows
        if "reasoning_tokens" in row
    }
    scored_keys = {
        f"{benchmark}-{example_id}"
        for benchmark, example_id in scores[["benchmark", "id"]].drop_duplicates().itertuples(index=False)
    }

    for benchmark in BENCHMARKS:
        response_meta = []
        for key, row in rows_by_key.items():
            if row["benchmark"] != benchmark or key not in scored_keys:
                continue
            response_meta.append(
                {
                    "key": key,
                    "id": str(row["id"]),
                    "label": _response_label(row),
                    "detail": _response_detail(row),
                    "correct": row.get("correct"),
                }
            )
        if not response_meta:
            continue
        response_meta.sort(
            key=lambda item: (
                0 if item["correct"] is True else 1 if item["correct"] is False else 2,
                item["id"],
            )
        )
        column_keys = [item["key"] for item in response_meta]
        manifest["benchmarks"].append(
            {"name": benchmark, "responses": response_meta, "column_keys": column_keys}
        )

        benchmark_scores = scores[scores.benchmark == benchmark].copy()
        benchmark_scores["response_key"] = benchmark_scores.apply(
            lambda row: f"{row['benchmark']}-{row['id']}", axis=1
        )
        for method_index, method in enumerate(methods):
            for layer_index, layer in enumerate(layers):
                subset = benchmark_scores[
                    (benchmark_scores.method == method) & (benchmark_scores.layer == layer)
                ]
                pivot = subset.pivot_table(
                    index="pair",
                    columns="response_key",
                    values="mean_cosine",
                    aggfunc="mean",
                )
                row_order = geometry[layer]["pair_order"]
                matrix = pivot.reindex(index=row_order, columns=column_keys).to_numpy(dtype=np.float32)
                positions = np.asarray([pair_position[pair] for pair in row_order], dtype=np.int64)
                standardized = _standardize(
                    matrix,
                    normalization["example_mean"][method_index, layer_index, positions],
                    normalization["example_std"][method_index, layer_index, positions],
                )

                raw_filename = f"probe-scenarios-{benchmark}-{method}-L{layer}-raw.json"
                z_filename = f"probe-scenarios-{benchmark}-{method}-L{layer}-z.json"
                _write_json(analysis_dir / raw_filename, _matrix_payload(matrix, metric="cosine"))
                _write_json(
                    analysis_dir / z_filename,
                    _matrix_payload(standardized, metric="z_score", color_limit=3.0),
                )
                key = f"{benchmark}:{method}:{layer}"
                manifest["scenario_matrices_raw"][key] = f"analysis/{raw_filename}"
                manifest["scenario_matrices"][key] = f"analysis/{z_filename}"

    _write_json(analysis_dir / "manifest.json", manifest)
    return manifest



def _recorded_axes(scores: pd.DataFrame) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return the method/layer rectangle with the broadest response coverage."""
    frame = scores[["benchmark", "id", "method", "layer"]].drop_duplicates().copy()
    if frame.empty:
        return (), ()
    frame["response_key"] = frame["benchmark"].astype(str) + "-" + frame["id"].astype(str)
    coverage = frame.groupby(["method", "layer"])["response_key"].nunique()
    maximum = int(coverage.max())
    complete = {(str(method), int(layer)) for (method, layer), count in coverage.items() if int(count) == maximum}
    methods = tuple(
        method
        for method in AVAILABLE_METHODS
        if any(candidate_method == method for candidate_method, _ in complete)
    )
    layers = tuple(
        layer
        for layer in AVAILABLE_LAYERS
        if any(candidate_layer == layer for _, candidate_layer in complete)
    )
    # Evaluator configurations are rectangular: every selected method is scored on
    # every selected layer. Keep only axes whose full Cartesian product has maximum coverage.
    methods = tuple(
        method for method in methods if all((method, layer) in complete for layer in layers)
    )
    layers = tuple(
        layer for layer in layers if all((method, layer) in complete for method in methods)
    )
    return methods, layers

def write_viewer(
    results: Path,
    response_rows: list[dict[str, Any]],
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
    correlations_table: pd.DataFrame,
) -> None:
    viewer = results / "concept_viewer"
    viewer.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "concept_viewer.html", viewer / "index.html")

    pair_ids = sorted(int(value) for value in scores.pair.dropna().unique())
    methods, layers = _recorded_axes(scores)
    if not methods or not layers:
        raise ValueError("Concept score files contain no supported methods or layers")
    normalization = _normalization_statistics(
        results, response_rows, scores, pair_ids, methods, layers
    )
    pair_position = {pair: index for index, pair in enumerate(pair_ids)}

    index: dict[str, Any] = {
        "pairs": pairs[
            ["pair", "concept", "antagonist", "class_name", "macro_class"]
        ].to_dict("records"),
        "responses": [],
        "methods": list(methods),
        "layers": list(layers),
        "correlations": json.loads(correlations_table.to_json(orient="records")),
        "analysis_manifest": "analysis/manifest.json",
        "normalization": {
            "default_scale": "standardized",
            "scope": "pooled recorded evaluation examples, reasoning tokens, and full generated responses",
            "dtype": "float16",
        },
    }
    for row in response_rows:
        if "reasoning_tokens" not in row:
            continue
        key = f"{row['benchmark']}-{row['id']}"
        response_scores = scores[
            (scores.benchmark == row["benchmark"]) & (scores.id == str(row["id"]))
        ]
        rankings = {}
        for (method, layer), group in response_scores.groupby(["method", "layer"]):
            method_index = methods.index(str(method))
            layer_index = layers.index(int(layer))
            items = []
            for record in group.sort_values("mean_cosine", ascending=False).itertuples(index=False):
                pair = int(record.pair)
                activation = float(record.mean_cosine)
                position = pair_position[pair]
                mean = float(normalization["example_mean"][method_index, layer_index, position])
                std = float(normalization["example_std"][method_index, layer_index, position])
                z_activation = (activation - mean) / std if std > 1e-6 else None
                items.append(
                    {
                        "pair": pair,
                        "activation": activation,
                        "z_activation": float(z_activation) if z_activation is not None else None,
                        "direction": "positive" if activation >= 0 else "negative",
                    }
                )
            rankings[f"{method}:{layer}"] = items
        _write_json(
            viewer / f"{key}.json",
            {
                "tokens": row["reasoning_tokens"],
                "rankings": rankings,
                "trace_root": f"../traces/{row['benchmark']}/{row['id']}",
                "dtype": "float16",
            },
        )
        index["responses"].append(
            {
                "key": key,
                "benchmark": row["benchmark"],
                "id": str(row["id"]),
                "correct": row.get("correct"),
                "reasoning_status": row.get("reasoning_status"),
            }
        )

    write_analysis_files(results, viewer, response_rows, scores, pairs, normalization, methods, layers)
    _write_json(viewer / "index.json", index)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    response_rows = [row for benchmark in BENCHMARKS for row in records(args.results, benchmark)]
    response = pd.DataFrame(response_rows)
    score_files = [args.results / f"concept_scores-{benchmark}.parquet" for benchmark in BENCHMARKS]
    score_files = [path for path in score_files if path.exists()]
    if response.empty:
        raise SystemExit("No evaluation results found. Run evaluate.py with --concept-analysis first.")

    response["id"] = response["id"].astype(str)
    if score_files:
        scores = pd.concat([pd.read_parquet(path) for path in score_files], ignore_index=True)
    else:
        scores = pd.DataFrame(
            columns=[
                "benchmark", "id", "method", "layer", "pair",
                "mean_cosine", "reasoning_tokens",
            ]
        )
    scores = _repair_score_coverage(args.results, response_rows, scores)
    if scores.empty:
        raise SystemExit("No compatible concept-analysis traces were found.")
    for benchmark in BENCHMARKS:
        benchmark_scores = scores[scores["benchmark"].astype(str) == benchmark]
        if benchmark_scores.empty:
            continue
        destination = args.results / f"concept_scores-{benchmark}.parquet"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        benchmark_scores.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(destination)
    scores["id"] = scores["id"].astype(str)
    scores["pair"] = scores["pair"].astype(int)
    scores["layer"] = scores["layer"].astype(int)
    methods, layers = _recorded_axes(scores)
    if not methods or not layers:
        raise SystemExit("Concept score files contain no consistently recorded method/layer configuration.")
    scores = scores[
        scores.method.astype(str).isin(methods) & scores.layer.astype(int).isin(layers)
    ].copy()
    scores = scores.merge(
        response[["benchmark", "id", "correct"]],
        on=["benchmark", "id"],
        how="left",
        validate="many_to_one",
    )

    pairs = pd.read_parquet(
        hf_hub_download(VECTOR_REPO, "pairs.parquet", revision=VECTOR_REVISION)
    )
    pairs["pair"] = pairs["pair"].astype(int)
    pairs["macro_class"] = pairs["class_name"].map(macro_class_name)
    correlations_table = correlations(scores).merge(
        pairs[["pair", "concept", "antagonist", "class_name"]],
        on="pair",
        how="left",
    )

    scores.to_parquet(args.results / "concept_scores.parquet", index=False)
    correlations_table.to_parquet(args.results / "correlations.parquet", index=False)
    correlations_table.to_csv(args.results / "correlations.csv", index=False)

    write_viewer(args.results, response_rows, scores, pairs, correlations_table)
    print(
        f"Wrote {args.results / 'correlations.parquet'} and "
        f"{args.results / 'concept_viewer' / 'index.html'}"
    )


if __name__ == "__main__":
    main()
