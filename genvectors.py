#! /usr/bin/env python

import json
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.numpy import save_file

log = logging.getLogger("genvectors")


def load(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one shard of sufficient statistics written by genstats.py.

    :param path: safetensors file holding the summed activations, with the run's configuration in
        its `manifest` metadata key.

    :return: the manifest, and the shard's tensors keyed by name.
    """
    with safe_open(str(path), framework="np") as handle:
        return json.loads(handle.metadata()["manifest"]), {key: handle.get_tensor(key) for key in handle.keys()}


def scatter(moment: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Recover the pooled within-class covariance from a second moment and the group sums.

    This is Proposition 2 of `$improve.tex`: with groups `g` of size `n_g` and sums `s_g`,
    `Sw = (1/N) (M - sum_g s_g s_g^T / n_g)`. The group sums alone cannot give it -- a sum of
    vectors says nothing about the sum of their outer products -- which is the whole reason `M` has
    to be accumulated during the forward pass rather than derived from the shards afterwards.

    Whitening by the *total* covariance instead would be a mistake: it decomposes as `Sw + Sb`, and
    with 1036 concept pairs the between-group term is large, so it would penalise exactly the
    directions that separate the poles.

    :param moment: `sum_i x_i x_i^T` over the stories counted here, `[hidden, hidden]`.
    :param sums: per-group sums over the same stories, `[group, hidden]`.
    :param counts: stories per group, `[group]`; empty groups are dropped rather than divided by.

    :return: the within-class covariance, `[hidden, hidden]`.
    """
    live = counts > 0
    return (moment - (sums[live] / counts[live, None]).T @ sums[live]) / counts.sum()


def fisher(direction: np.ndarray, delta: np.ndarray, within: np.ndarray) -> np.ndarray:
    """Score directions by the Fisher criterion `(w.delta)^2 / (w.Sw.w)`, per concept pair.

    :param direction: candidate directions, one per pair, `[pair, hidden]`.
    :param delta: the contrast each direction is meant to detect, `[pair, hidden]`.
    :param within: the within-class covariance the projection is measured against, `[hidden, hidden]`.

    :return: the criterion per pair, `[pair]`; larger separates the poles better.
    """
    # As one three-operand einsum this contracts pair-by-pair against the full 4096-square matrix
    # and runs orders of magnitude slower; going through the matmul first keeps it to one BLAS call.
    return np.einsum("ph,ph->p", direction, delta) ** 2 / np.einsum("ph,ph->p", direction @ within, direction)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    manifest: dict[str, Any] = {}
    totals: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for path in sorted(args.inputs):
        head, tensors = load(path)
        if head["shard"] in seen:
            raise SystemExit(f"{path} repeats shard {head['shard']}; the sums would be double counted")
        if manifest and head["config_hash"] != manifest["config_hash"]:
            raise SystemExit(f"{path} was produced by a different configuration; these are not summable")
        if manifest and set(tensors) != set(totals):
            raise SystemExit(
                f"{path} disagrees with the other shards on {sorted(set(tensors) ^ set(totals))}; "
                f"mixing shards from different script versions would leave those statistics "
                f"covering only part of the corpus with no further warning"
            )
        seen.add(head["shard"])
        manifest = manifest or head
        for key, value in tensors.items():
            totals[key] = totals[key] + value.astype(np.float64) if key in totals else value.astype(np.float64)
        log.info(f"shard {head['shard']}: {head['summary']['stories']} stories")

    if len(seen) != manifest["shards"]:
        raise SystemExit(
            f"only shards {sorted(seen)} of {manifest['shards']} given; a partial merge silently biases "
            f"the means, and because the corpus strides by pair_number a missing shard drops a whole variant"
        )

    layers, hidden, pairs_n = manifest["layers"], manifest["hidden_size"], manifest["n_pairs"]
    stories = int(totals["counts"].sum())
    log.info(f"merged {len(seen)} shards: {stories:,} stories over {int(totals['tokens'].sum()):,} tokens")

    ontology = json.loads(
        Path(hf_hub_download("AntonKorznikov/feature_stories", "ontology.json", repo_type="dataset")).read_text()
    )
    pairs = sorted(
        ({"class_name": entry["name"], **pair} for entry in ontology["classes"] for pair in entry["pairs"]),
        key=lambda pair: (pair["concept"], pair["antagonist"]),
    )
    if len(pairs) != pairs_n:
        raise SystemExit(f"ontology has {len(pairs)} pairs but the shards were built against {pairs_n}")

    per_pole = totals["counts"].sum(axis=2)
    if not (per_pole > 0).all():
        raise SystemExit(f"{int((per_pole == 0).sum())} (pair, pole) cells have no stories at all")
    mu = totals["sums"].sum(axis=3) / per_pole[None, :, :, None]
    mu_fold = totals["sums"] / totals["counts"][None, :, :, :, None]
    grand = totals["corpus_sum"] / stories

    vectors = {
        "diff": (mu[:, :, 0] - mu[:, :, 1]).astype(np.float32),
        "concept_centered": (mu[:, :, 0] - grand[:, None, :]).astype(np.float32),
        "antagonist_centered": (mu[:, :, 1] - grand[:, None, :]).astype(np.float32),
    }

    if "moments" in totals:
        # Shipped vectors use every story; the honest comparison below never does. One layer at a
        # time: the second moment is [2, layers, 4096, 4096] float64, and materialising more than a
        # couple of 4096-square matrices at once will not fit alongside the sums.
        flat = totals["sums"].reshape(len(layers), pairs_n * 2, 2, hidden)
        counts_flat = totals["counts"].reshape(pairs_n * 2, 2)
        whitened, verdict = np.empty_like(vectors["diff"]), []
        for position, layer in enumerate(layers):
            pooled = scatter(
                totals["moments"][:, position].sum(axis=0),
                flat[position].sum(axis=1),
                counts_flat.sum(axis=1),
            )
            ridge = args.shrinkage * np.trace(pooled) / hidden
            whitened[position] = np.linalg.solve(pooled + ridge * np.eye(hidden), vectors["diff"][position].T).T

            # Fit on one fold and score on the other, with both the contrast and the covariance
            # taken from the fitting fold only. $improve.tex §sec:verify pools Sw across folds and
            # calls the leak out; the per-fold moments remove it entirely.
            for fold in (0, 1):
                within = scatter(totals["moments"][fold, position], flat[position, :, fold], counts_flat[:, fold])
                held = scatter(
                    totals["moments"][1 - fold, position], flat[position, :, 1 - fold], counts_flat[:, 1 - fold]
                )
                fit = mu_fold[position, :, 0, fold] - mu_fold[position, :, 1, fold]
                test = mu_fold[position, :, 0, 1 - fold] - mu_fold[position, :, 1, 1 - fold]
                candidate = np.linalg.solve(
                    within + args.shrinkage * np.trace(within) / hidden * np.eye(hidden), fit.T
                ).T
                scored, baseline = fisher(candidate, test, held), fisher(fit, test, held)
                verdict.append(
                    {
                        "layer": layer,
                        "fit_fold": fold,
                        "wins": float((scored > baseline).mean()),
                        "median_ratio": float(np.median(scored / baseline)),
                    }
                )
            log.info(
                f"L{layer:02d}: whitened beats diff out of sample on "
                f"{100 * np.mean([v['wins'] for v in verdict[-2:]]):.1f}% of pairs, "
                f"median J ratio {np.mean([v['median_ratio'] for v in verdict[-2:]]):.2f}x"
            )
        vectors["lda"] = whitened.astype(np.float32)

    halves = mu_fold[:, :, 0] - mu_fold[:, :, 1]
    reliability = np.einsum("lpd,lpd->lp", halves[:, :, 0], halves[:, :, 1]) / (
        np.linalg.norm(halves[:, :, 0], axis=2) * np.linalg.norm(halves[:, :, 1], axis=2)
    )
    norms = np.linalg.norm(vectors["diff"], axis=2)
    label_rate = totals["hits"].reshape(pairs_n, 2, 2).sum(axis=(1, 2)) / per_pole.sum(axis=1)

    classes = np.array([pair["class_name"] for pair in pairs])
    same = classes[:, None] == classes[None, :]
    off = ~np.eye(pairs_n, dtype=bool)
    null_sigma = 1.0 / np.sqrt(hidden)

    diagnostics = {}
    for position, layer in enumerate(layers):
        unit = vectors["diff"][position] / norms[position][:, None]
        cosines = (unit @ unit.T).astype(np.float64)
        shared = vectors["diff"][position].mean(axis=0)
        # Eigenvalues of the pair-by-pair Gram matrix of the centred directions are the non-zero
        # spectrum of their covariance, and far cheaper than a hidden-by-hidden eigendecomposition.
        centred = unit - unit.mean(axis=0)
        spectrum = np.clip(np.linalg.eigvalsh(centred @ centred.T)[::-1], 0.0, None)
        participation = float(spectrum.sum() ** 2 / (spectrum**2).sum())
        diagnostics[f"L{layer:02d}"] = {
            "layer": layer,
            "depth": round(layer / manifest["n_model_layers"], 4),
            "grand_norm": float(np.linalg.norm(grand[position])),
            "diff_norm_median": float(np.median(norms[position])),
            "split_half_cos_median": float(np.median(reliability[position])),
            "reliable_pairs": int((reliability[position] > 0.5).sum()),
            "abs_cos_same_class": float(np.abs(cosines[same & off]).mean()),
            "abs_cos_other_class": float(np.abs(cosines[~same & off]).mean()),
            "cos_other_class_signed": float(cosines[~same & off].mean()),
            "anti_aligned_same_class": float((cosines[same & off] < 0).mean()),
            "shared_component": float(np.median(np.abs(unit @ (shared / np.linalg.norm(shared))))),
            "effective_dim": participation,
            "components_half_variance": int(np.searchsorted(np.cumsum(spectrum) / spectrum.sum(), 0.5) + 1),
            "chance_abs_cos_in_subspace": float(np.sqrt(2.0 / (np.pi * participation))),
        }
        entry = diagnostics[f"L{layer:02d}"]
        log.info(
            f"L{layer:02d} (depth {entry['depth']}): split-half {entry['split_half_cos_median']:.3f}, "
            f"{entry['reliable_pairs']}/{pairs_n} pairs above 0.5"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    run = {
        "model": manifest["model"],
        "n_model_layers": manifest["n_model_layers"],
        "layers": layers,
        "hidden_size": hidden,
        "n_pairs": pairs_n,
        "skip_tokens": manifest["skip_tokens"],
        "rendered_prefix": manifest["rendered_prefix"],
        "config_hash": manifest["config_hash"],
        "shards_merged": sorted(seen),
        "stories": stories,
        "tokens": int(totals["tokens"].sum()),
        "nonfinite": int(totals["dropped"].sum()),
        "axes": ["layer", "pair", "hidden"],
        "null_cosine_sigma": null_sigma,
        "layer_names": [f"L{layer:02d}" for layer in layers],
        "rows": "pairs.parquet",
        "diagnostics": diagnostics,
    }
    for name, tensor in vectors.items():
        save_file({name: tensor}, str(args.out / f"{name}.safetensors"), metadata={"manifest": json.dumps(run)})

    table = {
        "pair": pa.array(range(pairs_n), pa.int32()),
        "concept": pa.array([pair["concept"] for pair in pairs]),
        "antagonist": pa.array([pair["antagonist"] for pair in pairs]),
        "class_name": pa.array([pair["class_name"] for pair in pairs]),
        "narrative_guidance": pa.array([pair["narrative_guidance"] for pair in pairs]),
        "n_concept": pa.array(per_pole[:, 0].astype("int64")),
        "n_antagonist": pa.array(per_pole[:, 1].astype("int64")),
        "label_rate": pa.array(label_rate),
    }
    for position, layer in enumerate(layers):
        table[f"L{layer:02d}_diff_norm"] = pa.array(norms[position])
        table[f"L{layer:02d}_rel_norm"] = pa.array(norms[position] / diagnostics[f"L{layer:02d}"]["grand_norm"])
        table[f"L{layer:02d}_split_half_cos"] = pa.array(reliability[position])
    pq.write_table(pa.table(table), args.out / "pairs.parquet")

    geometry_table = "\n".join(
        f"| {entry['layer']} | {entry['effective_dim']:.1f} | {entry['abs_cos_same_class']:.3f} | "
        f"{entry['abs_cos_other_class']:.3f} | {entry['chance_abs_cos_in_subspace']:.3f} | "
        f"{entry['abs_cos_same_class'] / entry['abs_cos_other_class']:.2f} | "
        f"{100 * entry['anti_aligned_same_class']:.0f}% |"
        for entry in diagnostics.values()
    )
    layer_table = "\n".join(
        f"| {entry['layer']} | {entry['depth']} | {entry['split_half_cos_median']:.3f} | "
        f"{entry['reliable_pairs']} | {entry['diff_norm_median']:.1f} |"
        for entry in diagnostics.values()
    )
    named = ", ".join(str(layer) for layer in layers[:-1]) + f", and {layers[-1]}"
    (args.out / "README.md").write_text(f"""---
license: mit
base_model: {manifest["model"]}
tags:
  - interpretability
  - alignment
  - steering
  - concept-vectors
---

# Behavioral concept vectors for `{manifest["model"]}`
{pairs_n} contrastive behavioral directions read off the residual stream of `{manifest["model"]}`,
extracted from [`AntonKorznikov/feature_stories`](https://huggingface.co/datasets/AntonKorznikov/feature_stories)
by difference of means over matched concept/antagonist story pairs.

## Files
| File | Method |
| - | - |
| `diff.safetensors` | `mean(concept) - mean(antagonist)` |
| `concept_centered.safetensors` | `mean(concept) - mean(corpus)` |
| `antagonist_centered.safetensors` | `mean(antagonist) - mean(corpus)` |

We analyze layers {named} of {manifest["n_model_layers"]}.

`pairs.parquet` has one row per pair **in the same order as the pair axis**,
so row `i` names and scores tensor row `i`.
It also carries the concept and antagonist labels,
the ontology class,
the narrative guidance,
the story counts behind each pole,
and per layer the norm and split-half reliability.

`diff` subtracts the antipole,
which this corpus supports because both poles of a row describe the same premise, so topic,
genre and language cancel exactly.
The two `*_centered` tensors subtract the corpus mean.
The following equality holds:
`diff == concept_centered - antagonist_centered`.

Pooling is the mean over story tokens from position {manifest["skip_tokens"]} onward.
Each story is presented as an assistant turn behind a fixed, concept-free user prompt,
with no system prompt.

## Layers
| Layer | Depth | Median Split-Half Cosine | Pairs Above 0.5 | Median Diff Norm |
| - | - | - | - | - |
{layer_table}

## Geometry
| Layer | Same Class | Different Class | Ratio | Anti-Aligned | Shared Component |
| - | - | - | - | - | - |
{geometry_table}

Values are absolute cosines between the `diff` directions of two different pairs,
split by whether the pairs belong to the same ontology class.
For random unit vectors in {hidden} dimensions the expected absolute cosine is {null_sigma * (2 / np.pi) ** 0.5:.4f}.
""")
    log.info(f"wrote {args.out}")

    if repo := os.getenv("HF_REPO"):
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=str(args.out),
            commit_message=f"Concept vectors from {manifest['model']} at blocks {layers}",
        )
        log.info(f"published https://huggingface.co/{repo}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="shard files written by genstats.py")
    parser.add_argument("--out", type=Path, required=True, help="directory to write the vectors into")
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.05,
        help="ridge fraction blended into the within-class covariance before inverting it; the "
        "effective-sample-size argument for whitening only holds post-shrinkage",
    )
    main(parser.parse_args())
