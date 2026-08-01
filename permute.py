#! /usr/bin/env python

"""Build a permuted-label null: difference vectors between mismatched pairs.

The 512 isotropic directions used so far answer "does this concept beat an arbitrary direction in
R^4096?". They do not answer "does it beat a direction built the same way from the same corpus?",
and the gap between those matters: at block 25 a `diff` vector carries 71% of its energy in the top
ten principal directions of the concept set, against 0.25% for an isotropic direction. Some of the
measured dose response is therefore attributable to *where these vectors point* rather than to what
they mean.

This null closes that. For a real vector, `v_i = mean(concept_i) - mean(antagonist_i)`. For a
permuted one, `v_ij = mean(concept_i) - mean(antagonist_j)` with `j != i`: the same pooling over the
same stories at the same layers, differing only in that the two poles no longer belong to the same
premise. Norm, subspace alignment and every property of the extraction survive; only the semantic
pairing is destroyed.

Correctness is checked rather than asserted: with `j == i` the construction must reproduce the
published `diff.safetensors` to floating-point tolerance. If it does not, the pooling here differs
from the pooling that made the vectors and the null would not be comparable.

Reads `genstats.py` shards, which carry per-(pair, pole, fold) activation sums and counts. No GPU
and no model: a difference of means is arithmetic on numbers already computed.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

log = logging.getLogger("permute")


def totals(shards: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Sum per-group activation sums and counts across every shard.

    :param shards: `shard.safetensors` files written by `genstats.py`.

    :return: sums `[layer, pair, side, hidden]` and counts `[pair, side]`, folds already collapsed.
    """
    sums = counts = None
    for shard in shards:
        with safe_open(str(shard), framework="np") as handle:
            block = np.asarray(handle.get_tensor("sums"), dtype=np.float64).sum(axis=3)
            tally = np.asarray(handle.get_tensor("counts"), dtype=np.int64).sum(axis=2)
        sums = block if sums is None else sums + block
        counts = tally if counts is None else counts + tally
    return sums, counts


def poles(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-pole mean activation.

    :param sums: `[layer, pair, side, hidden]`.
    :param counts: `[pair, side]`.

    :return: `[layer, pair, side, hidden]` means, zero where a group had no stories.
    """
    safe = np.maximum(counts, 1)[None, :, :, None]
    return np.where(counts[None, :, :, None] > 0, sums / safe, 0.0)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    shards = sorted(args.moments.glob("shard-*/shard.safetensors"))
    if not shards:
        raise SystemExit(f"no shard-*/shard.safetensors under {args.moments}")
    sums, counts = totals(shards)
    means = poles(sums, counts)
    layers, pairs, _, hidden = means.shape
    log.info(f"{len(shards)} shards, {pairs} pairs, {layers} layers, {hidden} dims")
    log.info(f"stories per pole: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    # The identity permutation must reproduce the published vectors, or the pooling differs.
    rebuilt = means[:, :, 0] - means[:, :, 1]
    published = load_file(args.probes / "diff.safetensors")["diff"].astype(np.float64)
    scale = np.linalg.norm(published, axis=-1)
    error = np.linalg.norm(rebuilt - published, axis=-1) / np.maximum(scale, 1e-12)
    cosine = np.einsum("lph,lph->lp", rebuilt, published) / np.maximum(
        np.linalg.norm(rebuilt, axis=-1) * scale, 1e-12)
    log.info(f"identity check: median cosine {np.median(cosine):.8f}, worst {cosine.min():.8f}, "
             f"max relative error {error.max():.2e}")
    if cosine.min() < args.tolerance:
        raise SystemExit(f"rebuild does not reproduce diff.safetensors: worst cosine {cosine.min():.6f}")

    # Mismatched pairs, sampled without j == i. Pairs whose either pole is empty are skipped, since a
    # zero mean would make a direction that is really just the other pole.
    live = np.flatnonzero((counts > 0).all(axis=1))
    log.info(f"{len(live)} pairs have stories on both poles")
    generator = np.random.default_rng(args.seed)
    left = generator.choice(live, size=args.controls, replace=True)
    right = generator.choice(live, size=args.controls, replace=True)
    clash = left == right
    while clash.any():
        right[clash] = generator.choice(live, size=int(clash.sum()), replace=True)
        clash = left == right
    permuted = means[:, left, 0] - means[:, right, 1]
    log.info(f"{args.controls} mismatched directions, {len(set(zip(left.tolist(), right.tolist())))} distinct")

    real_norm = np.linalg.norm(rebuilt, axis=-1)
    fake_norm = np.linalg.norm(permuted, axis=-1)
    log.info(f"norms: real median {np.median(real_norm):.3f}, permuted median {np.median(fake_norm):.3f}")

    unit = permuted / np.maximum(np.linalg.norm(permuted, axis=-1, keepdims=True), 1e-12)
    axis = rebuilt / np.maximum(np.linalg.norm(rebuilt, axis=-1, keepdims=True), 1e-12)
    for slot in range(layers):
        _, _, basis = np.linalg.svd(axis[slot] - axis[slot].mean(0), full_matrices=False)
        share_real = ((axis[slot] @ basis[:10].T) ** 2).sum(axis=1).mean()
        share_fake = ((unit[slot] @ basis[:10].T) ** 2).sum(axis=1).mean()
        log.info(f"layer slot {slot}: energy in the concept top-10 PCs -- real {share_real:.3f}, "
                 f"permuted {share_fake:.3f}")

    save_file(
        {"diff": permuted.astype(np.float32)},
        str(args.out),
        metadata={
            "manifest": json.dumps({
                "kind": "permuted-label null",
                "construction": "mean(concept_i) - mean(antagonist_j), j != i",
                "pairs_available": int(len(live)), "controls": int(args.controls),
                "seed": int(args.seed), "shards": len(shards),
                "identity_check_min_cosine": float(cosine.min()),
                "layers": [11, 14, 18, 22, 25],
            }),
        },
    )
    log.info(f"wrote {args.out}: {permuted.shape} f32, {permuted.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--moments", type=Path, default=Path(".bak/stats"))
    parser.add_argument("--probes", type=Path, default=Path(".bak/probes"))
    parser.add_argument("--out", type=Path, default=Path("permuted.safetensors"))
    parser.add_argument("--controls", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--tolerance", type=float, default=0.9999)
    main(parser.parse_args())
