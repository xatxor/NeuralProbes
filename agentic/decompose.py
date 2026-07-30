#! /usr/bin/env python

"""Ask what an extracted direction is made of, before spending a GPU-hour steering with it.

Three questions, in increasing order of how much they can embarrass the vector:

1. **Is it the length axis?** The cheapest thing that separates hacked from gave-up trajectories in
   this corpus is episode shape -- four scalars reach AUC 0.939 on their own. If the extracted vector
   is nearly parallel to a direction fitted on length alone, Corollary 2 already told us what
   happened and no steering run is needed.
2. **Does it live inside the concept span?** Projecting it onto the 1036 published directions splits
   it into a part the existing concepts can express and a residual they cannot. A small residual
   means the direction is an affect/semantic readout we could have found passively; a large one means
   there is a component of the decision those concepts miss, which is the interesting case.
3. **Which concepts, and is that above chance?** With effective dimensionality near ten, a cosine of
   0.3 to some concept out of 1036 is unremarkable. The null here is not zero, so it is measured:
   random unit vectors are pushed through the identical pipeline and their best-of-1036 cosine is the
   bar the real vector has to clear.

Nothing here touches a GPU or a model; it is arithmetic on saved vectors and runs in seconds.
"""

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file

log = logging.getLogger("decompose")

VECTORS = "josephofthebread/Qwen3-8B-concept-vectors"
# The published file's manifest records layers [11, 14, 18, 22, 25]; agent.py steers with the same
# row mapping, so a mismatch here would compare a vector against concepts from another depth.
ROW = {11: 0, 14: 1, 18: 2, 22: 3, 25: 4}


def find(name: str) -> str:
    """Locate a file inside the local HF cache.

    :param name: the file's name within the vectors repo.

    :return: an absolute path.
    """
    pattern = f"**/models--{VECTORS.replace('/', '--')}/snapshots/*/{name}"
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / pattern), recursive=True)
    if not matches:
        raise SystemExit(f"{name} is not in the local cache; this box is meant to have it already")
    return matches[0]


def explained(vector: np.ndarray, basis: np.ndarray) -> tuple[float, np.ndarray]:
    """Fraction of a vector's squared norm that the span of a basis can express.

    The basis is rank-deficient by construction -- 1036 directions whose effective dimensionality is
    about ten -- so this goes through an SVD rather than a normal-equations solve, which would be
    solving a singular system and reporting whatever the conditioning produced.

    :param vector: unit direction to decompose.
    :param basis: `[concept, hidden]` directions.

    :return: the explained fraction and the residual direction.
    """
    # Orthonormal basis for the row space, keeping only directions with real support.
    _, singular, right = np.linalg.svd(basis, full_matrices=False)
    keep = right[singular > singular[0] * 1e-6]
    inside = keep.T @ (keep @ vector)
    residual = vector - inside
    return float(inside @ inside), residual


def null(basis: np.ndarray, draws: int, rng: np.random.Generator) -> dict:
    """What a random direction achieves against the same 1036 concepts.

    :param basis: `[concept, hidden]` unit directions.
    :param draws: random vectors to try.
    :param rng: source of randomness.

    :return: percentiles of best-of-1036 absolute cosine, and of explained fraction.
    """
    noise = rng.normal(size=(draws, basis.shape[1]))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    best = np.abs(noise @ basis.T).max(axis=1)
    _, singular, right = np.linalg.svd(basis, full_matrices=False)
    keep = right[singular > singular[0] * 1e-6]
    inside = (noise @ keep.T)
    return {
        "best_cosine_p50": float(np.quantile(best, 0.50)),
        "best_cosine_p95": float(np.quantile(best, 0.95)),
        "best_cosine_max": float(best.max()),
        "explained_p50": float(np.quantile((inside * inside).sum(axis=1), 0.50)),
        "explained_p95": float(np.quantile((inside * inside).sum(axis=1), 0.95)),
        "rank": int(keep.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector", type=Path, required=True, help="unit direction as .npy")
    parser.add_argument("--against", type=Path, action="append", default=[],
                        help="other saved directions to report a cosine against; repeatable")
    parser.add_argument("--layer", type=int, default=18, choices=sorted(ROW))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    v = np.load(args.vector).astype(np.float64).flatten()
    v /= np.linalg.norm(v)

    block = load_file(find("diff.safetensors"))["diff"]
    basis = block[ROW[args.layer]].float().numpy().astype(np.float64)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    rows = pq.read_table(find("pairs.parquet")).to_pylist()

    cosines = basis @ v
    order = np.argsort(-np.abs(cosines))[: args.top]
    share, residual = explained(v, basis)
    reference = null(basis, args.draws, rng)

    report = {
        "vector": str(args.vector),
        "layer": args.layer,
        "explained_by_concept_span": share,
        "residual_norm": float(np.linalg.norm(residual)),
        "null": reference,
        "top": [
            {
                "pair": int(i),
                "cosine": float(cosines[i]),
                "concept": rows[i]["concept"],
                "antagonist": rows[i]["antagonist"],
                "class": rows[i]["class_name"],
            }
            for i in order
        ],
        "against": {},
    }

    for other in args.against:
        w = np.load(other).astype(np.float64).flatten()
        w /= np.linalg.norm(w)
        report["against"][Path(other).stem] = float(v @ w)

    log.info(f"{args.vector.name} at L{args.layer}")
    log.info(f"  concept span (rank {reference['rank']}) explains {share:.3f} of it; "
             f"a random direction gets {reference['explained_p50']:.3f} (p95 {reference['explained_p95']:.3f})")
    log.info(f"  best |cos| to any of 1036: {abs(cosines[order[0]]):.3f}; "
             f"random gets {reference['best_cosine_p50']:.3f} (p95 {reference['best_cosine_p95']:.3f})")
    for entry in report["top"][:8]:
        flag = "" if abs(entry["cosine"]) > reference["best_cosine_p95"] else "   (inside the null)"
        log.info(f"  {entry['cosine']:+.3f}  {entry['pair']:>4}  "
                 f"{entry['concept']} || {entry['antagonist']}{flag}")
    for name, value in report["against"].items():
        log.info(f"  cosine to {name}: {value:+.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
