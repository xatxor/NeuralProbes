#! /usr/bin/env python

"""Is the dose response selective, or does most of the ontology move?

`dose-figures.pdf` figure 3 shows the six concepts that scored best. That is exactly the figure most
likely to mislead, because it is chosen by the statistic it illustrates. This script draws the same
curves at fixed *ranks* through the sorted list instead -- 1st, 100th, 500th, 1000th -- so the
spectrum is visible rather than its top. A row of random control directions is drawn on the same
axes as the floor.

Two things are deliberately held constant across every panel: the y-limits, so a weak response looks
weak, and the set of ladders, so the magnitude control is visible everywhere.

The per-class table at the end asks the same question through the ontology instead of through the
ranking: if `Crisis & Safety` classes move and `Aesthetics` classes do not, the response is about
danger. If every class moves about the same amount, it is about something the whole ontology shares.
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

log = logging.getLogger("dosecompare")

LAYERS = [11, 14, 18, 22, 25]
LADDERS = ("tylenol", "syrup", "ibuprofen", "steps")
COLOURS = ("C3", "C1", "C4", "C0")
COSINE = 0


def curve(ax: Any, blob: Any, index: dict[str, Any], render: str, slot: int, column: int,
          title: str, ylim: float) -> None:
    """Draw one concept's dose-response across all four ladders.

    :param ax: target axes.
    :param blob: the loaded npz.
    :param index: manifest keyed by `ladder.rendering`.
    :param render: which rendering to read.
    :param slot: index along the layer axis.
    :param column: index along the direction axis.
    :param title: panel title.
    :param ylim: symmetric y-limit shared by every panel.
    """
    for ladder, colour in zip(LADDERS, COLOURS):
        key = f"{ladder}.{render}"
        series = blob[f"{key}.values"][:, COSINE, -1, slot, column]
        ax.plot(index[key]["doses"], series - series[0], marker="o", ms=2.5, lw=1.1,
                color=colour, label=ladder)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-ylim, ylim)
    ax.set_title(title, fontsize=6.5)
    ax.tick_params(labelsize=5.5)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    blob = np.load(args.readout, allow_pickle=False)
    manifest = json.loads(str(blob["manifest"]))
    meta = json.loads(str(blob["meta"]))
    index = {f"{e['ladder']}.{e['rendering']}": e for e in manifest}
    concepts = meta["concepts"]

    table = pd.read_parquet(args.table)
    layer = args.layer
    slot = LAYERS.index(layer)
    block = table[table.layer == layer].reset_index(drop=True)
    order = block.z_tylenol.abs().sort_values(ascending=False).index.to_numpy()

    ylim = float(np.abs(
        blob[f"tylenol.{args.rendering}.values"][:, COSINE, -1, slot, :concepts]
        - blob[f"tylenol.{args.rendering}.values"][0, COSINE, -1, slot, :concepts]
    ).max()) * 1.05
    log.info(f"block {layer}: shared y-limit +/-{ylim:.4f}")

    rank_rows = [
        ("rank 1-6 of 1036", [0, 1, 2, 3, 4, 5]),
        ("rank 50, 100, 150, 200, 250, 300", [49, 99, 149, 199, 249, 299]),
        ("rank 400, 500, 600, 700, 850, 1000", [399, 499, 599, 699, 849, 999]),
    ]

    with PdfPages(args.out) as pdf:
        fig, axes = plt.subplots(4, 6, figsize=(13.5, 9.2))
        for row, (label, ranks) in enumerate(rank_rows):
            for col, rank in enumerate(ranks):
                entry = block.loc[order[rank]]
                curve(axes[row, col], blob, index, args.rendering, slot, int(entry.pair),
                      f"#{rank + 1}  {entry.concept[:30]}\nz={entry.z_tylenol:.1f}  "
                      f"steps={entry.z_steps:.1f}", ylim)
            axes[row, 0].set_ylabel(f"{label}\n" r"$\Delta\cos(\hat v,\hat h)$ vs lowest rung", fontsize=6)

        generator = np.random.default_rng(args.seed)
        picks = generator.choice(meta["controls"], size=6, replace=False)
        for col, pick in enumerate(picks):
            curve(axes[3, col], blob, index, args.rendering, slot, concepts + int(pick),
                  f"random direction #{pick}", ylim)
        axes[3, 0].set_ylabel("512 random directions\n" r"$\Delta\cos(\hat v,\hat h)$ vs lowest rung", fontsize=6)
        axes[0, 0].legend(fontsize=5)

        fig.suptitle(
            f"Dose-response at the final token, block {layer}, by rank rather than by cherry-pick.  "
            "Same y-scale in every panel.", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # The ontology asks the same question a different way.
        classes = (block.assign(absz=block.z_tylenol.abs(), leak=block.z_steps.abs())
                   .groupby("class_name")
                   .agg(n=("absz", "size"), mean_z=("absz", "mean"), mean_leak=("leak", "mean"))
                   .query("n >= @args.min_class")
                   .sort_values("mean_z", ascending=False))

        fig, ax = plt.subplots(figsize=(9, 8.5))
        shown = pd.concat([classes.head(args.classes), classes.tail(args.classes)])
        colours = ["C3"] * min(args.classes, len(classes)) + ["C0"] * min(args.classes, len(classes))
        ax.barh(range(len(shown)), shown.mean_z, color=colours[:len(shown)])
        ax.barh(range(len(shown)), shown.mean_leak, height=0.35, color="k", alpha=0.5,
                label="mean |z| on the steps control")
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([f"{i}  (n={int(r.n)})" for i, r in shown.iterrows()], fontsize=6.5)
        ax.invert_yaxis()
        ax.axvline(float(block.z_tylenol.abs().median()), color="k", ls="--", lw=1,
                   label=f"ontology median = {block.z_tylenol.abs().median():.1f}")
        ax.set_xlabel(r"mean $|z|$ vs Tylenol dose")
        ax.set_title(f"Ontology classes, block {layer}: strongest {args.classes} and weakest "
                     f"{args.classes} of {len(classes)}", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    spread = classes.mean_z
    log.info(f"wrote {args.out}")
    log.info(f"classes: {len(classes)}, mean |z| from {spread.min():.2f} to {spread.max():.2f}, "
             f"ontology median {block.z_tylenol.abs().median():.2f}")
    print(classes.head(12).round(2).to_string())
    print("...")
    print(classes.tail(8).round(2).to_string())


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, default=Path("dose-readout.npz"))
    parser.add_argument("--table", type=Path, default=Path("dose-correlations.parquet"))
    parser.add_argument("--out", type=Path, default=Path("dose-spectrum.pdf"))
    parser.add_argument("--rendering", default="chat", choices=["chat", "raw"])
    parser.add_argument("--layer", type=int, default=25, choices=LAYERS)
    parser.add_argument("--classes", type=int, default=18)
    parser.add_argument("--min-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    main(parser.parse_args())
