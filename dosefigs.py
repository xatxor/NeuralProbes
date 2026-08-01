#! /usr/bin/env python

"""Every figure in the dose-response report, from `doseall.py`'s derived arrays.

One plotting implementation, one page per figure, in the order the paper uses them.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

log = logging.getLogger("dosefigs")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
CONTROL = "steps"
LADDERS = DANGER + (CONTROL,)
COLOURS = {"tylenol": "C3", "syrup": "C1", "ibuprofen": "C4", "steps": "C0"}


def clean(token: str) -> str:
    """Make one Qwen token printable on an axis.

    :param token: raw byte-level BPE token.

    :return: a printable form.
    """
    text = token.replace("Ġ", " ").replace("Ċ", "\\n")
    return text if text.strip() else text.replace(" ", "_")


def peakseries(values: np.ndarray, argmax: np.ndarray, slot: int, column: int) -> np.ndarray:
    """One direction's value across rungs, read at its own peak token.

    :param values: `[rung, token, layer, column]`.
    :param argmax: `[layer, column]` peak token index.
    :param slot: layer index.
    :param column: direction index.

    :return: `[rung]`.
    """
    return values[:, int(argmax[slot, column]), slot, column]


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    blob = np.load(args.derived, allow_pickle=False)
    summary = json.loads(args.summary.read_text())
    table = pd.read_parquet(args.stats)
    meta = summary["meta"]
    concepts, controls = meta["concepts"], meta["controls"]
    tokens = [clean(t) for t in json.loads(str(blob["tokens"]))]
    doses = json.loads(str(blob["doses"]))
    prednames = json.loads(str(blob["prednames"]))
    real = slice(0, concepts)
    slot25 = LAYERS.index(25)

    with PdfPages(args.out) as pdf:
        # 1 --- what the model does, in text length -------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
        lengths = summary["reply_tokens"]
        keys = sorted(lengths, key=int)
        axes[0].plot([int(k) for k in keys], [np.mean(lengths[k]) for k in keys],
                     marker="o", color="C3")
        for k in keys:
            axes[0].scatter([int(k)] * len(lengths[k]), lengths[k], s=6, color="C3", alpha=0.4)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Tylenol dose (mg)")
        axes[0].set_ylabel("reply length (tokens)")
        axes[0].set_title("(a) reply length, 5 continuations per rung", fontsize=9)
        counts = [summary["per_layer_final"][str(l)] for l in LAYERS]
        width = 0.2
        for offset, ladder in enumerate(LADDERS):
            axes[1].bar(np.arange(len(LAYERS)) + offset * width, [c[ladder] for c in counts],
                        width, color=COLOURS[ladder], label=ladder)
        axes[1].set_xticks(np.arange(len(LAYERS)) + 1.5 * width)
        axes[1].set_xticklabels([str(n) for n in LAYERS])
        axes[1].set_xlabel("block")
        axes[1].set_ylabel(r"concepts with $|z|\geq4$")
        axes[1].set_title("(b) prompt side, final token", fontsize=9)
        axes[1].legend(fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2 --- which estimator is reproducible --------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        names = list(summary["estimators"])
        pairs_shown = ["tylenol-syrup", "tylenol-ibuprofen", "syrup-ibuprofen", "tylenol-steps"]
        width = 0.2
        for offset, key in enumerate(pairs_shown):
            colour = "C0" if key.endswith("steps") else "C3"
            axes[0].bar(np.arange(len(names)) + offset * width,
                        [summary["estimators"][n]["25"][key] for n in names], width,
                        color=colour, alpha=1.0 if offset == 0 else 0.45 + 0.18 * offset,
                        label=key)
        axes[0].axhline(0, color="k", lw=0.6)
        axes[0].set_xticks(np.arange(len(names)) + 1.5 * width)
        axes[0].set_xticklabels(names, fontsize=7)
        axes[0].set_ylabel(r"agreement $r$ over 1036 concepts")
        axes[0].set_title("(a) block 25: does the estimator transport between ladders?", fontsize=8.5)
        axes[0].legend(fontsize=6)
        for offset, ladder in enumerate(LADDERS):
            axes[1].bar(np.arange(len(names)) + offset * width,
                        [summary["estimator_counts"][n]["25"][ladder] for n in names], width,
                        color=COLOURS[ladder], label=ladder)
        axes[1].set_xticks(np.arange(len(names)) + 1.5 * width)
        axes[1].set_xticklabels(names, fontsize=7)
        axes[1].set_ylabel(r"concepts with $|z|\geq4$")
        axes[1].set_title("(b) block 25: how many respond", fontsize=8.5)
        axes[1].legend(fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3 --- every readout position, both criteria -------------------------------------------
        sweep = summary["sweep"]
        names = [n for n in sweep if n.startswith("token")]
        names.sort(key=lambda n: int(n.split()[1]), reverse=True)
        extra = ["content mean", "max"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
        index = np.arange(len(names))
        for ax, key, caption in zip(
                axes, ("dd", "separation"),
                ("(a) agreement among the three danger ladders",
                 "(b) that agreement minus agreement with the control")):
            for slot, layer in enumerate(LAYERS):
                ax.plot(index, [sweep[n][str(layer)][key] for n in names], lw=1.1, marker="o",
                        ms=2.5, label=f"block {layer}")
            for offset, name in enumerate(extra):
                ax.scatter([len(names) + offset], [sweep[name]["25"][key]], marker="s", s=26,
                           color="k", zorder=5)
                ax.annotate(name, (len(names) + offset, sweep[name]["25"][key]), fontsize=6,
                            rotation=90, ha="center", va="bottom")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(list(index[::2]) + [len(names), len(names) + 1])
            ax.set_xticklabels([n.replace("token ", "") for n in names[::2]] + ["mean", "max"],
                               fontsize=5.5, rotation=90)
            ax.set_xlabel("prompt position, counted from the end", fontsize=8)
            ax.set_title(caption, fontsize=8.5)
            ax.tick_params(labelsize=6)
        axes[0].set_ylabel(r"correlation over 1036 concepts", fontsize=8)
        axes[0].legend(fontsize=6, ncol=5)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4 --- per-token map ------------------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
        ceiling = 1.0
        for ax, ladder, caption in zip(axes, ("tylenol", CONTROL),
                                       ("Tylenol dose", "step count (control)")):
            values, shared = blob[f"pv.{ladder}"], blob[f"sh.{ladder}"]
            delta = np.abs(values[-1] - values[0])[:, :, real]
            control = np.abs(values[-1] - values[0])[:, :, concepts:]
            z = delta / np.maximum(control.std(axis=-1, keepdims=True), 1e-12)
            # A fixed ceiling saturated: the peak runs past 40, so most of the map rendered as one
            # flat colour. The scale is taken from the danger ladder and reused for the control, so
            # the two panels stay comparable.
            grid = z.max(axis=-1).T
            ceiling = ceiling if ladder != "tylenol" else float(np.percentile(grid, 99))
            im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=ceiling,
                           interpolation="nearest")
            ax.set_yticks(range(len(LAYERS)))
            ax.set_yticklabels([str(n) for n in LAYERS], fontsize=7)
            ax.set_ylabel("block", fontsize=8)
            for position in np.where(~shared)[0]:
                ax.axvline(position, color="cyan", lw=1.2)
            ax.set_title(rf"$\max_c |z|$, {caption}; cyan = the token that differs", fontsize=9)
            fig.colorbar(im, ax=ax, pad=0.01)
        axes[-1].set_xticks(range(len(tokens)))
        axes[-1].set_xticklabels(tokens, rotation=90, fontsize=5, family="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 5 --- where the peak lands ------------------------------------------------------------
        fig, axes = plt.subplots(1, len(LAYERS), figsize=(13, 2.5), sharey=True)
        width_tokens = blob["pv.tylenol"].shape[1]
        for slot, (ax, layer) in enumerate(zip(axes, LAYERS)):
            ax.hist(blob["am.tylenol"][slot, real] - width_tokens,
                    bins=np.arange(-width_tokens, 1) - 0.5, color="C3")
            ax.set_title(f"block {layer}", fontsize=8)
            ax.set_xlabel("peak token, from end", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[0].set_ylabel("concepts", fontsize=7)
        fig.suptitle("peak position of the identical-token response", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 6 --- concepts against the measured null ----------------------------------------------
        fig, axes = plt.subplots(2, len(LAYERS), figsize=(13, 4.6), sharey="row")
        for slot, layer in enumerate(LAYERS):
            for row, (source, caption) in enumerate((("zf.tylenol", "prompt"), ("zr.tylenol", "reply"))):
                z = blob[source][slot]
                top = np.percentile(np.abs(z[real]), 99.5)
                axes[row, slot].hist(np.abs(z[concepts:]), bins=30, range=(0, top), density=True,
                                     color="0.75", label=f"{controls} random")
                axes[row, slot].hist(np.abs(z[real]), bins=30, range=(0, top), density=True,
                                     histtype="step", color="C3", lw=1.3, label=f"{concepts} concepts")
                axes[row, slot].axvline(4, color="k", ls="--", lw=0.8)
                axes[row, slot].tick_params(labelsize=6)
                axes[row, slot].set_xlabel(rf"$|z|$, {caption}", fontsize=7)
            axes[0, slot].set_title(f"block {layer}", fontsize=8)
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 7 --- the spectrum, by rank -----------------------------------------------------------
        values, argmax = blob["pv.tylenol"], blob["am.tylenol"]
        order = np.argsort(-np.abs(blob["zf.tylenol"][slot25, real]))
        ylim = 1.05 * float(np.abs(values[:, -1, slot25, real]
                                   - values[0:1, -1, slot25, real]).max())
        rows = [("rank 1-6", [0, 1, 2, 3, 4, 5]),
                ("rank 50-300", [49, 99, 149, 199, 249, 299]),
                ("rank 400-1000", [399, 499, 599, 699, 849, 999])]
        fig, axes = plt.subplots(4, 6, figsize=(13, 8.4))
        for row, (caption, ranks) in enumerate(rows):
            for col, rank in enumerate(ranks):
                pair = int(order[rank])
                ax = axes[row, col]
                for ladder in LADDERS:
                    series = blob[f"pv.{ladder}"][:, -1, slot25, pair]
                    ax.plot(doses[ladder], series - series[0], marker="o", ms=2.5, lw=1.1,
                            color=COLOURS[ladder], label=ladder)
                ax.set_xscale("log")
                ax.axhline(0, color="k", lw=0.5)
                ax.set_ylim(-ylim, ylim)
                ax.set_title(f"#{rank + 1} {str(table.iloc[pair].concept)[:26]}\n"
                             f"z={blob['zf.tylenol'][slot25, pair]:.1f} "
                             f"steps={blob['zf.steps'][slot25, pair]:.1f}", fontsize=6)
                ax.tick_params(labelsize=5)
            axes[row, 0].set_ylabel(f"{caption}\n" r"$\Delta\cos$, final token", fontsize=6)
        picks = np.random.default_rng(args.seed).choice(controls, size=6, replace=False)
        for col, pick in enumerate(picks):
            ax = axes[3, col]
            for ladder in LADDERS:
                series = blob[f"pv.{ladder}"][:, -1, slot25, concepts + int(pick)]
                ax.plot(doses[ladder], series - series[0], marker="o", ms=2.5, lw=1.1,
                        color=COLOURS[ladder])
            ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylim(-ylim, ylim)
            ax.set_title(f"random #{pick}", fontsize=6)
            ax.tick_params(labelsize=5)
        axes[3, 0].set_ylabel("random directions\n" r"$\Delta\cos$, final token", fontsize=6)
        axes[0, 0].legend(fontsize=5)
        fig.suptitle("block 25, prompt side, ranked by this block's own statistic; shared y-scale",
                     fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 8 --- reply side, with continuation spread --------------------------------------------
        rorder = np.argsort(-np.abs(blob["zr.tylenol"][slot25, real]))
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.2))
        for ax, rank in zip(axes.ravel(), range(8)):
            pair = int(rorder[rank])
            for ladder in LADDERS:
                mean = blob[f"rv.{ladder}"][:, slot25, pair]
                each = blob[f"rs.{ladder}"][:, :, slot25, pair]
                ax.plot(doses[ladder], mean - mean[0], marker="o", ms=3, lw=1.2,
                        color=COLOURS[ladder], label=ladder)
                ax.fill_between(doses[ladder], (each - mean[0:1, None]).min(axis=1),
                                (each - mean[0:1, None]).max(axis=1),
                                color=COLOURS[ladder], alpha=0.15, lw=0)
            ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{str(table.iloc[pair].concept)[:28]}\n"
                         f"reply z={blob['zr.tylenol'][slot25, pair]:.1f}", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[0, 0].set_ylabel(r"$\Delta\cos$, reply mean", fontsize=7)
        axes[1, 0].set_ylabel(r"$\Delta\cos$, reply mean", fontsize=7)
        axes[0, 0].legend(fontsize=5.5)
        fig.suptitle("block 25, reply side; band spans the five continuations", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 9 --- which prompt position predicts the reply ----------------------------------------
        fig, ax = plt.subplots(figsize=(9, 3.4))
        excess, base = [], []
        for label in prednames:
            row = summary["prediction"][label]["25"]
            excess.append(row["concepts_mean_abs_r"] - row["controls_mean_abs_r"])
            base.append(row["controls_mean_abs_r"])
        index = np.arange(len(prednames))
        ax.bar(index, base, 0.6, color="0.75", label="random directions")
        ax.bar(index, excess, 0.6, bottom=base, color="C3", label="excess over random")
        ax.set_xticks(index)
        ax.set_xticklabels(prednames, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(r"mean $|r|$ with reply content")
        ax.set_title("block 25: prompt position against the reply it precedes", fontsize=9)
        ax.legend(fontsize=7)
        for i, value in enumerate(excess):
            ax.text(i, base[i] + value + 0.01, f"{value:+.3f}", ha="center", fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 10 --- danger against magnitude ---------------------------------------------------------
        fig, axes = plt.subplots(2, len(LAYERS), figsize=(13, 5.0), sharex="row", sharey="row")
        for slot, layer in enumerate(LAYERS):
            for row, prefix in enumerate(("zf", "zr")):
                lim = float(np.percentile(np.abs(blob[f"{prefix}.tylenol"][slot, real]), 99.5))
                axes[row, slot].scatter(blob[f"{prefix}.steps"][slot, concepts:],
                                        blob[f"{prefix}.tylenol"][slot, concepts:], s=3, color="0.75")
                axes[row, slot].scatter(blob[f"{prefix}.steps"][slot, real],
                                        blob[f"{prefix}.tylenol"][slot, real], s=3, color="C3",
                                        alpha=0.45)
                axes[row, slot].plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":")
                axes[row, slot].set_xlim(-lim, lim)
                axes[row, slot].set_ylim(-lim, lim)
                axes[row, slot].tick_params(labelsize=6)
            axes[0, slot].set_title(f"block {layer}", fontsize=8)
            axes[1, slot].set_xlabel(r"$z$, step count", fontsize=7)
        axes[0, 0].set_ylabel(r"$z$ Tylenol, prompt", fontsize=7)
        axes[1, 0].set_ylabel(r"$z$ Tylenol, reply", fontsize=7)
        fig.suptitle("points on the diagonal track the number, not the danger", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 11 --- ontology classes -----------------------------------------------------------------
        block = table[table.layer == 25]
        classes = (block.assign(a=block.f_tylenol.abs(), b=block.f_steps.abs())
                   .groupby("class_name").agg(n=("a", "size"), z=("a", "mean"), leak=("b", "mean"))
                   .query("n >= 4").sort_values("z", ascending=False))
        shown = pd.concat([classes.head(15), classes.tail(15)])
        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        ax.barh(range(len(shown)), shown.z, color=["C3"] * 15 + ["C0"] * 15)
        ax.barh(range(len(shown)), shown.leak, height=0.35, color="k", alpha=0.5, label="step control")
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([f"{i} (n={int(r.n)})" for i, r in shown.iterrows()], fontsize=6.5)
        ax.invert_yaxis()
        ax.axvline(float(block.f_tylenol.abs().median()), color="k", ls="--", lw=1,
                   label=f"median {block.f_tylenol.abs().median():.1f}")
        ax.set_xlabel(r"mean $|z|$, block 25, prompt side")
        ax.set_title(f"strongest and weakest 15 of {len(classes)} ontology classes", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    log.info(f"wrote {args.out}, 11 figures")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, default=Path("dose-derived.npz"))
    parser.add_argument("--summary", type=Path, default=Path("dose-summary.json"))
    parser.add_argument("--stats", type=Path, default=Path("dose-stats.parquet"))
    parser.add_argument("--out", type=Path, default=Path("dose-figures.pdf"))
    parser.add_argument("--seed", type=int, default=7)
    main(parser.parse_args())
