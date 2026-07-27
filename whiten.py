#! /usr/bin/env python

import json
import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

log = logging.getLogger("whiten")


def scatter(moment: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pooled within-class covariance from second moments and per-group sums.

    `Sw = (1/N)(M - sum_g s_g s_g^T / n_g)`, Proposition 2 of `$improve.tex`.

    :param moment: uncentred second moment summed over every story, as `[hidden, hidden]`.
    :param sums: per-group activation sums, as `[group, hidden]`.
    :param counts: stories behind each group, as `[group]`.

    :return: the pooled within-group covariance, `[hidden, hidden]`.
    """
    live = counts > 0
    return (moment - (sums[live] / counts[live, None]).T @ sums[live]) / counts.sum()


def totals(shards: list[Path], position: int, hidden: int, groups: int) -> tuple[np.ndarray, ...]:
    """Sum one layer's second moments and group statistics across every shard.

    The full `moments` tensor is `[2, 5, 4096, 4096]` float64, 1.3 GB per shard, so it is sliced by
    layer on the way in rather than loaded and indexed: `get_slice` reads only the bytes asked for.

    :param shards: `shard.safetensors` files written by `genstats.py`.
    :param position: index of the wanted layer within the layer axis.
    :param hidden: residual width.
    :param groups: number of (pair, pole) groups, twice the pair count.

    :return: the summed second moment, per-group sums, and per-group counts.
    """
    moment = np.zeros((hidden, hidden), dtype=np.float64)
    sums, counts = np.zeros((groups, hidden), dtype=np.float64), np.zeros(groups, dtype=np.int64)
    for shard in shards:
        with safe_open(str(shard), framework="np") as handle:
            moment += np.asarray(handle.get_slice("moments")[:, position], dtype=np.float64).sum(axis=0)
            sums += (
                np.asarray(handle.get_slice("sums")[position], dtype=np.float64).reshape(groups, 2, hidden).sum(axis=1)
            )
            counts += handle.get_tensor("counts").reshape(groups, 2).sum(axis=1)
    return moment, sums, counts


def load(path: Path, name: str) -> np.ndarray:
    """Read one whole `[layer, pair, hidden]` tensor out of a published safetensors file.

    :param path: the file to read.
    :param name: tensor key, which is always the file stem in this repository's artifacts.

    :return: the tensor as float64.
    """
    with safe_open(str(path), framework="np") as handle:
        return np.asarray(handle.get_tensor(name), dtype=np.float64)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()

    shards = sorted(args.moments.glob("shard-*/shard.safetensors"))
    if not shards:
        raise SystemExit(f"no shard-*/shard.safetensors under {args.moments}")

    plain = ("diff", "concept_centered", "antagonist_centered")
    vectors = {name: np.asarray(load(args.probes / f"{name}.safetensors", name)) for name in plain}
    layers, pairs, hidden = vectors["diff"].shape
    log.info(f"{len(shards)} moment shards, {pairs} pairs at {hidden} dims, {layers} layers")

    stacked = np.zeros((2 * len(plain), layers, pairs, hidden), dtype=np.float32)
    for index, name in enumerate(plain):
        stacked[index] = vectors[name]

    for position in range(layers):
        moment, sums, counts = totals(shards, position, hidden, 2 * pairs)
        within = scatter(moment, sums, counts)
        # A trace-proportional ridge at the shrinkage genvectors.py used, so `whitened_diff` reproduces
        # the published `lda` exactly rather than approximately.
        ridge = 0.05 * np.trace(within) / hidden
        solved = np.linalg.solve(within + ridge * np.eye(hidden), np.hstack([vectors[n][position].T for n in plain]))
        for index, name in enumerate(plain):
            stacked[len(plain) + index, position] = solved[:, index * pairs : (index + 1) * pairs].T
        log.info(f"layer {position}: solved, {time.monotonic() - started:.0f}s elapsed")

    # `whitened_diff` is by construction the published `lda`. If they disagree, either the moments or
    # the shrinkage moved since publication, and every whitened readout below would be built on sand.
    if (reference := args.probes / "lda.safetensors").exists():
        published = np.asarray(load(reference, "lda"))
        ours = stacked[len(plain)]
        cosine = np.einsum("lph,lph->lp", ours, published) / (
            np.linalg.norm(ours, axis=2) * np.linalg.norm(published, axis=2)
        )
        log.info(f"whitened_diff vs published lda: median cosine {np.median(cosine):.6f}, worst {cosine.min():.6f}")
        if cosine.min() < 0.999:
            raise SystemExit(f"whitened_diff does not reproduce lda: worst cosine {cosine.min():.6f}")

    save_file(
        {"readouts": stacked},
        str(args.out),
        metadata={
            "axes": json.dumps(["method", "layer", "pair", "hidden"]),
            "methods": json.dumps(list(plain) + [f"whitened_{name}" for name in plain]),
            "layers": json.dumps([11, 14, 18, 22, 25]),
            "shrinkage": "0.05",
            "shards": str(len(shards)),
        },
    )
    log.info(
        f"wrote {args.out}: {stacked.shape} f32, {stacked.nbytes / 1e6:.0f} MB, "
        f"{(time.monotonic() - started) / 60:.1f}m"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--probes", type=Path, default=Path("probes-lda"), help="directory of published vectors")
    parser.add_argument("--moments", type=Path, default=Path("moments"), help="directory of genstats.py shards")
    parser.add_argument("--out", type=Path, default=Path("readouts.safetensors"))
    main(parser.parse_args())
