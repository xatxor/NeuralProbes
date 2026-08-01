#! /usr/bin/env python

"""Relate an extracted direction to every published vector family.

`whiten.py` emits six readouts over the same 1036 concepts: `diff` (difference of the two poles),
`concept_centered` and `antagonist_centered` (each pole against the global mean, the one-vs-all
constructions), and a whitened counterpart of each, obtained by solving against the pooled
within-class covariance with a trace-proportional ridge. They are six different answers to "which
direction represents this concept", and they disagree, so asking which of them an outcome-supervised
direction lands in is a real question rather than a formality.

Two quantities per family, and only the second is worth much on its own.

**Best single cosine** says whether any one named concept aligns with the extracted direction. It is
reported against a measured null, because with 1036 correlated directions in 4096 dimensions the
best-of-1036 cosine is substantially above zero for a random vector -- quoting it against 0 would
manufacture a finding.

**Explained fraction** says how much of the direction the family's whole span can express. This is the
quantity Fable's residual experiment turns on: a small residual means the extracted direction is
essentially a combination of concepts defined without ever seeing a reward hack, and a large one
means there is a component of the decision that no named concept reaches. The residual is written out
so it can itself be steered and ablated, which is the only way to find out whether the unnamed part
does any work.

The span is rank-deficient -- 1036 directions whose effective dimensionality is about ten -- so the
projection goes through an SVD rather than a normal-equations solve, which would be inverting a
singular matrix and reporting the conditioning.
"""

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

log = logging.getLogger("families")

VECTORS = "josephofthebread/Qwen3-8B-concept-vectors"
LAYERS = (11, 14, 18, 22, 25)


def find(name: str) -> str:
    """Locate a file inside the local HF cache.

    :param name: the file's name within the vectors repo.

    :return: an absolute path.
    """
    pattern = f"**/models--{VECTORS.replace('/', '--')}/snapshots/*/{name}"
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / pattern), recursive=True)
    if not matches:
        raise SystemExit(f"{name} is not in the local cache")
    return matches[0]


def analyse(vector: np.ndarray, basis: np.ndarray, draws: int, rng: np.random.Generator) -> dict:
    """Cosines, span coverage, and the matching null for one family.

    :param vector: unit direction under test.
    :param basis: `[concept, hidden]` directions for one family at one layer.
    :param draws: random directions for the null.
    :param rng: source of randomness.

    :return: the measured quantities and what chance achieves.
    """
    unit = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    _, singular, right = np.linalg.svd(unit, full_matrices=False)
    span = right[singular > singular[0] * 1e-6]

    cosines = unit @ vector
    inside = span.T @ (span @ vector)
    residual = vector - inside

    noise = rng.normal(size=(draws, unit.shape[1]))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    null_best = np.abs(noise @ unit.T).max(axis=1)
    null_span = ((noise @ span.T) ** 2).sum(axis=1)

    return {
        "rank": int(span.shape[0]),
        "best_cosine": float(np.abs(cosines).max()),
        "best_cosine_null_p95": float(np.quantile(null_best, 0.95)),
        "explained": float(inside @ inside),
        "explained_null_p95": float(np.quantile(null_span, 0.95)),
        "residual": residual,
        "cosines": cosines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector", type=Path, action="append", required=True,
                        help="extracted direction as .npy; repeatable")
    parser.add_argument("--readouts", type=Path, default=Path("readouts.safetensors"))
    parser.add_argument("--layer", type=int, default=18, choices=LAYERS)
    parser.add_argument("--out", type=Path, default=Path("analysis/families.json"))
    parser.add_argument("--residuals", type=Path, default=Path("bipo"))
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    held = load_file(args.readouts)
    stacked = held["readouts"]
    methods = ["diff", "concept_centered", "antagonist_centered",
               "whitened_diff", "whitened_concept_centered", "whitened_antagonist_centered"]
    if stacked.shape[0] != len(methods):
        raise SystemExit(f"expected {len(methods)} methods in {args.readouts}, found {stacked.shape[0]}")
    position = LAYERS.index(args.layer)
    rows = pq.read_table(find("pairs.parquet")).to_pylist()

    report: dict = {"layer": args.layer, "methods": methods, "vectors": {}}
    for path in args.vector:
        v = np.load(path).astype(np.float64).flatten()
        v /= np.linalg.norm(v)
        log.info(f"=== {path.name} at L{args.layer} ===")
        entry: dict = {}
        for index, method in enumerate(methods):
            got = analyse(v, stacked[index, position].astype(np.float64), args.draws, rng)
            cosines = got.pop("cosines")
            residual = got.pop("residual")
            order = np.argsort(-np.abs(cosines))[: args.top]
            got["top"] = [
                {"pair": int(j), "cosine": float(cosines[j]),
                 "concept": rows[j]["concept"], "antagonist": rows[j]["antagonist"],
                 "class": rows[j]["class_name"]}
                for j in order
            ]
            entry[method] = got

            norm = np.linalg.norm(residual)
            if norm > 1e-9:
                args.residuals.mkdir(parents=True, exist_ok=True)
                target = args.residuals / f"{path.stem}-residual-{method}-L{args.layer}.npy"
                np.save(target, (residual / norm).astype(np.float32))

            flag = "" if got["best_cosine"] > got["best_cosine_null_p95"] else "  (inside null)"
            log.info(f"  {method:<32} rank {got['rank']:>4}  best|cos| {got['best_cosine']:.3f} "
                     f"(null {got['best_cosine_null_p95']:.3f}){flag}  span explains "
                     f"{got['explained']:.3f} (null {got['explained_null_p95']:.3f})")
            for row in got["top"][:3]:
                log.info(f"      {row['cosine']:+.3f}  {row['pair']:>4}  {row['concept']} || {row['antagonist']}")
        report["vectors"][path.stem] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
