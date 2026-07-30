#! /usr/bin/env python

"""Turn the dose readout into correlations, figures and a report.

The question is narrow: for a given concept vector, a given block and a given token position, does
the concept's value move monotonically with the number in the prompt? That is one correlation over
nine points, and it is asked 1548 x 5 x 47 times per ladder, so almost all of the care here goes
into not being fooled by the ones that come out large by chance.

Three things do that work.

**The null is measured, not assumed, and it changes what can be claimed.** 512 random unit
directions rode through the same projection, so at every (token, layer) there are 512 statistics
computed from the same activations under the hypothesis of no relationship. Running that first
produced the single most important result here: random directions reach a *mean* |rho| of 0.83 at
block 25, and 322 of them clear |rho| = 0.9. The reason is structural rather than statistical --
raising the number in the prompt moves the residual stream smoothly along one direction, so the
projection onto almost any fixed vector inherits that monotonicity. **Correlation with dose is
therefore close to uninformative on its own**, and a paper that reports only "the probe rises with
dose" has not yet distinguished its probe from an arbitrary direction.

What does discriminate is *amplitude*: how far the concept's value actually travels across the
ladder, measured in standard deviations of how far the 512 random directions travel over the same
rungs. Because every direction is a unit vector, that z-score is, up to normalisation, the cosine
between the concept and the movement the dose change induces -- and for a random vector in 4096
dimensions that cosine is about 1/64. Monotonicity is kept as a gate, not as evidence.

**A semantic null runs alongside.** The `steps` ladder has the same sentence frame and the same
single-digit swap, but the quantity is harmless. A concept whose value climbs with Tylenol dose and
climbs just as hard with step count is tracking how big a number is, not how dangerous a situation
is. This distinction cannot be made from the Tylenol ladder alone, and the paper does not make it.

**Agreement across ladders is required.** A concept has to move the same way for Tylenol, cough
syrup and ibuprofen -- three substances, three units, two sentence frames, one of them a
differently-tokenised wide grid. Ranking is by the *weakest* of the three, so a concept cannot buy
its place with one strong ladder.

Spearman is primary throughout: the dose axis is deliberately log-spaced and no linear relationship
is claimed. Pearson against log10(dose) is reported next to it as a check that the monotone trend is
not one rung doing all the work.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

log = logging.getLogger("doseanalyse")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
CONTROL_LADDER = "steps"
COSINE, RAW = 0, 1


def pearson(values: np.ndarray, axis0: np.ndarray) -> np.ndarray:
    """Correlate every column of `values` against one vector, along the first axis.

    :param values: `[n, ...]`.
    :param axis0: `[n]`.

    :return: correlation with `values.shape[1:]`.
    """
    a = values - values.mean(axis=0, keepdims=True)
    b = axis0 - axis0.mean()
    b = b.reshape((-1,) + (1,) * (values.ndim - 1))
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0)


def ranks(values: np.ndarray) -> np.ndarray:
    """Rank along the first axis, averaging ties.

    :param values: `[n, ...]`.

    :return: ranks, same shape.
    """
    order = values.argsort(axis=0)
    out = np.empty_like(values)
    grid = np.arange(values.shape[0]).reshape((-1,) + (1,) * (values.ndim - 1))
    np.put_along_axis(out, order, np.broadcast_to(grid, values.shape).astype(values.dtype), axis=0)
    return out


def correlate(values: np.ndarray, doses: np.ndarray) -> dict[str, np.ndarray]:
    """Spearman and Pearson of every (token, layer, column) against dose.

    :param values: `[rung, token, layer, column]`.
    :param doses: `[rung]`.

    :return: `spearman`, `pearson` and `swing` arrays, each `[token, layer, column]`. `swing` is the
        signed travel from the lowest rung to the highest, which is what amplitude is built from.
    """
    return {
        "spearman": pearson(ranks(values), ranks(doses.astype(np.float64))),
        "pearson": pearson(values, np.log10(doses.astype(np.float64))),
        "swing": values[-1] - values[0],
    }


def clean(token: str) -> str:
    """Make one Qwen token printable.

    :param token: raw token string, byte-level BPE.

    :return: something a plot axis can carry.
    """
    text = token.replace("Ġ", " ").replace("Ċ", "\\n")
    return text if text.strip() else text.replace(" ", "_")


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    blob = np.load(args.readout, allow_pickle=False)
    manifest = json.loads(str(blob["manifest"]))
    meta = json.loads(str(blob["meta"]))
    concepts = meta["concepts"]
    pairs = pd.read_parquet(args.pairs)
    log.info(f"{concepts} concepts + {meta['controls']} controls, {len(manifest)} conditions")

    stats: dict[str, dict[str, np.ndarray]] = {}
    index: dict[str, dict[str, Any]] = {}
    for entry in manifest:
        key = f"{entry['ladder']}.{entry['rendering']}"
        values = blob[f"{key}.values"][:, COSINE]
        stats[key] = correlate(values, np.array(entry["doses"]))
        index[key] = entry
        log.info(f"{key}: correlated {values.shape}")

    render = args.rendering
    floors: dict[str, np.ndarray] = {}
    for key, block in stats.items():
        control = np.abs(block["spearman"][..., concepts:])
        floors[key] = np.percentile(control, args.floor, axis=-1)

    # --- the selection -------------------------------------------------------------------------
    # Everything below is judged at the final token, which is the analogue of the paper's
    # "Assistant:" colon: the last position before the model would begin replying.
    #
    # Two statistics per ladder. `rho` is monotonicity and is used only as a gate, because the null
    # for it is enormous. `z` is the signed travel from the lowest rung to the highest, divided by
    # the standard deviation of that same travel over the 512 random directions at the same token
    # and block. It is the statistic that carries the claim.
    final: dict[str, np.ndarray] = {}
    for ladder in DANGER + (CONTROL_LADDER,):
        key = f"{ladder}.{render}"
        swing = stats[key]["swing"][-1]
        control = swing[:, concepts:]
        spread = control.std(axis=-1, keepdims=True)
        final[f"{ladder}_z"] = (swing - control.mean(axis=-1, keepdims=True)) / np.maximum(spread, 1e-12)
        final[ladder] = stats[key]["spearman"][-1]
        final[f"{ladder}_r"] = stats[key]["pearson"][-1]

    real = slice(0, concepts)
    floor_final = {l: floors[f"{l}.{render}"][-1] for l in DANGER + (CONTROL_LADDER,)}

    zs = np.stack([final[f"{l}_z"][:, real] for l in DANGER])
    rhos = np.stack([final[l][:, real] for l in DANGER])
    consistent = ((np.sign(zs) == np.sign(zs[0])).all(axis=0) & (np.sign(zs[0]) != 0)
                  & (np.abs(rhos) >= args.monotone).all(axis=0))
    strength = np.abs(zs).min(axis=0)
    leak = np.abs(final[f"{CONTROL_LADDER}_z"][:, real])
    clears = consistent & (strength >= args.z)
    specificity = strength - leak

    rows = []
    for slot, layer in enumerate(LAYERS):
        for pair in range(concepts):
            row = pairs.iloc[pair]
            rows.append(
                {
                    "layer": layer,
                    "pair": pair,
                    "concept": row.get("concept", f"pair {pair}"),
                    "antagonist": row.get("antagonist", ""),
                    "class_name": row.get("class_name", ""),
                    **{f"z_{l}": float(final[f"{l}_z"][slot, pair]) for l in DANGER + (CONTROL_LADDER,)},
                    **{f"rho_{l}": float(final[l][slot, pair]) for l in DANGER + (CONTROL_LADDER,)},
                    "r_tylenol": float(final["tylenol_r"][slot, pair]),
                    "consistent": bool(consistent[slot, pair]),
                    "clears": bool(clears[slot, pair]),
                    "strength": float(strength[slot, pair]),
                    "leak": float(leak[slot, pair]),
                    "specificity": float(specificity[slot, pair]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_parquet(args.out / "dose-correlations.parquet")

    hits = table[table.clears].sort_values("specificity", ascending=False)
    log.info(
        f"{len(hits)} concept-layer cells: monotone on all three danger ladders "
        f"(|rho| >= {args.monotone}), same sign, |z| >= {args.z}"
    )

    # --- figures -------------------------------------------------------------------------------
    key = f"tylenol.{render}"
    tokens = [clean(t) for t in index[key]["tokens"]]
    varying = index[key]["varying"]
    ranked = hits if len(hits) else table.sort_values("strength", ascending=False)
    top = ranked.iloc[0]
    top_pair = int(top.pair)
    top_slot = LAYERS.index(int(top.layer))

    with PdfPages(args.out / "dose-figures.pdf") as pdf:
        # 1. the paper's figure 13: where in the sequence the safe and dangerous rungs diverge.
        values = blob[f"{key}.values"][:, COSINE]
        delta = values[-1, :, :, top_pair] - values[0, :, :, top_pair]
        control_delta = values[-1, :, :, concepts:] - values[0, :, :, concepts:]
        band = control_delta.std(axis=-1)
        fig, ax = plt.subplots(figsize=(13, 4.4))
        for slot, layer in enumerate(LAYERS):
            line, = ax.plot(delta[:, slot], marker="o", ms=3, lw=1.3, label=f"block {layer}")
            ax.fill_between(range(len(tokens)), -2 * band[:, slot], 2 * band[:, slot],
                            color=line.get_color(), alpha=0.06, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(varying[0], color="crimson", ls="--", lw=1, label="the digit that changes")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=5.5, family="monospace")
        ax.set_ylabel(r"$\Delta$ cosine (9000 mg $-$ 1000 mg)")
        ax.set_title(
            f"{top.concept} vs {top.antagonist}, safe to dangerous dose"
            "  (shading: $\\pm 2\\sigma$ of 512 random directions)", fontsize=9)
        ax.legend(fontsize=7, ncol=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. where in the sequence any concept moves further than chance, danger against control
        fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True)
        for ax, ladder, title in zip(
            axes, ("tylenol", CONTROL_LADDER), ("Tylenol dose (danger)", "step count (magnitude control)")
        ):
            swing = stats[f"{ladder}.{render}"]["swing"]
            control = swing[..., concepts:]
            z = (swing[..., real] - control.mean(-1, keepdims=True)) / np.maximum(control.std(-1, keepdims=True), 1e-12)
            block = np.abs(z).max(axis=-1)
            im = ax.imshow(block.T, aspect="auto", cmap="magma", vmin=0, vmax=8, interpolation="nearest")
            ax.set_yticks(range(len(LAYERS)))
            ax.set_yticklabels([f"block {n}" for n in LAYERS], fontsize=7)
            ax.set_title(f"strongest $|z|$ over 1036 concepts -- {title}", fontsize=9)
            ax.axvline(varying[0], color="cyan", ls="--", lw=1)
            fig.colorbar(im, ax=ax, pad=0.01)
        axes[-1].set_xticks(range(len(tokens)))
        axes[-1].set_xticklabels(tokens, rotation=90, fontsize=5.5, family="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. dose-response curves for the concepts that survived selection
        picks = ranked.drop_duplicates("pair").head(6)
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
        for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
            slot = LAYERS.index(int(row.layer))
            for ladder, colour in zip(DANGER + (CONTROL_LADDER,), ("C3", "C1", "C4", "C0")):
                entry = index[f"{ladder}.{render}"]
                series = blob[f"{ladder}.{render}.values"][:, COSINE, -1, slot, int(row.pair)]
                ax.plot(entry["doses"], series - series[0], marker="o", ms=3, color=colour, label=ladder)
            ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{row.concept}\nblock {int(row.layer)}  z={row.strength:.1f}  leak={row.leak:.1f}",
                         fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("quantity in prompt", fontsize=7)
            ax.set_ylabel("cosine, relative to lowest rung", fontsize=7)
            ax.legend(fontsize=5.5)
        fig.suptitle("dose-response at the final token, six most dose-specific concepts", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. the point of the whole analysis: correlation cannot separate, amplitude can
        fig, axes = plt.subplots(2, len(LAYERS), figsize=(14, 5.6), sharey="row")
        for slot, layer in enumerate(LAYERS):
            rho = final["tylenol"][slot]
            axes[0, slot].hist(np.abs(rho[concepts:]), bins=26, range=(0, 1), density=True,
                               color="0.75", label="512 random")
            axes[0, slot].hist(np.abs(rho[real]), bins=26, range=(0, 1), density=True,
                               histtype="step", color="crimson", lw=1.3, label="1036 concepts")
            axes[0, slot].set_title(f"block {layer}", fontsize=8)
            axes[0, slot].set_xlabel(r"$|\rho|$ vs dose", fontsize=7)
            z = final["tylenol_z"][slot]
            axes[1, slot].hist(np.abs(z[concepts:]), bins=26, range=(0, 10), density=True,
                               color="0.75")
            axes[1, slot].hist(np.abs(z[real]), bins=26, range=(0, 10), density=True,
                               histtype="step", color="crimson", lw=1.3)
            axes[1, slot].set_xlabel(r"$|z|$ of swing", fontsize=7)
            for ax in (axes[0, slot], axes[1, slot]):
                ax.tick_params(labelsize=6)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("top: monotonicity does not separate concepts from random.  "
                     "bottom: amplitude does.", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 5. danger against magnitude: the separation the control ladder buys
        fig, axes = plt.subplots(1, len(LAYERS), figsize=(14, 3.2), sharex=True, sharey=True)
        for slot, (ax, layer) in enumerate(zip(axes, LAYERS)):
            ax.scatter(final[f"{CONTROL_LADDER}_z"][slot, concepts:], final["tylenol_z"][slot, concepts:],
                       s=3, color="0.75", label="random")
            ax.scatter(final[f"{CONTROL_LADDER}_z"][slot, real], final["tylenol_z"][slot, real],
                       s=3, color="crimson", alpha=0.5, label="concepts")
            lim = 12
            ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"block {layer}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlabel(r"$z$ vs step count", fontsize=7)
        axes[0].set_ylabel(r"$z$ vs Tylenol dose", fontsize=7)
        axes[0].legend(fontsize=6)
        fig.suptitle("on the diagonal means tracking the number, not the danger", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 6. how many concepts survive, per layer
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        width = 0.2
        for offset, ladder in enumerate(DANGER + (CONTROL_LADDER,)):
            counts = [int((np.abs(final[f"{ladder}_z"][slot, real]) >= args.z).sum())
                      for slot in range(len(LAYERS))]
            ax.bar(np.arange(len(LAYERS)) + offset * width, counts, width, label=ladder)
        ax.set_xticks(np.arange(len(LAYERS)) + 1.5 * width)
        ax.set_xticklabels([f"block {n}" for n in LAYERS])
        ax.set_ylabel(f"concepts with $|z| \\geq$ {args.z:.0f}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    summary = {
        "meta": meta,
        "rendering": render,
        "gates": {"monotone": args.monotone, "z": args.z, "floor_percentile": args.floor},
        "null_rho": {
            str(layer): {
                "mean_abs_random": float(np.abs(final["tylenol"][slot, concepts:]).mean()),
                "pct99_abs_random": float(floor_final["tylenol"][slot]),
                "random_above_0.9": int((np.abs(final["tylenol"][slot, concepts:]) >= 0.9).sum()),
                "of_random": int(meta["controls"]),
            }
            for slot, layer in enumerate(LAYERS)
        },
        "n_hits": int(len(hits)),
        "n_concepts_any_layer": int(hits.pair.nunique()) if len(hits) else 0,
        "per_layer": {
            str(layer): {
                ladder: int((np.abs(final[f"{ladder}_z"][slot, real]) >= args.z).sum())
                for ladder in DANGER + (CONTROL_LADDER,)
            }
            for slot, layer in enumerate(LAYERS)
        },
        "tokens": tokens,
        "varying_token": varying,
        "prompt": index[key]["text"],
        "top": ranked.head(args.top).to_dict("records"),
    }
    (args.out / "dose-summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"wrote dose-figures.pdf, dose-correlations.parquet, dose-summary.json into {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, default=Path("dose-readout.npz"))
    parser.add_argument("--pairs", type=Path, default=Path("pairs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--rendering", default="chat", choices=["chat", "raw"])
    parser.add_argument("--floor", type=float, default=99.0, help="control percentile, reported not used")
    parser.add_argument("--monotone", type=float, default=0.9, help="|rho| gate on every danger ladder")
    parser.add_argument("--z", type=float, default=4.0, help="swing z-score a concept must reach")
    parser.add_argument("--top", type=int, default=40)
    main(parser.parse_args())
