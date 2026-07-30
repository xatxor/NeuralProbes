#! /usr/bin/env python

"""The same spectrum figure, drawn at three different readout positions.

Where a probe is read is a free parameter that the original report fixes without discussion, and
which turned out to change the answer here more than any other choice. Three positions are drawn,
all inside the chat rendering so nothing else varies:

`prompt mean`   the average over the user's own sentence, tokens 3 to `<|im_end|>`, with every
                template token excluded. No special token contributes.

`assistant hdr` the last token of `<|im_start|>assistant\\n`. This is the structural counterpart of
                the paper's "Assistant:" -- the string that opens the assistant turn -- and it sits
                *before* Qwen3's forced-empty `<think>\\n\\n</think>` is appended.

`final token`   the `\\n\\n` that closes that empty reasoning block: the last position before the
                model emits its first real token. This is what the earlier report used, and it
                matches the paper's stated *definition* ("the last token before the Assistant's
                response") rather than the paper's literal colon.

Each page ranks concepts by that position's own statistic, so each shows what that position
surfaces. The bottom row of every page is six random directions on the same axes, and the y-limit is
shared across all three pages so the panels are comparable between them as well as within.
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

log = logging.getLogger("dosepositions")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
LADDERS = DANGER + ("steps",)
COLOURS = ("C3", "C1", "C4", "C0")
COSINE = 0

# name -> (payload suffix, token index). `None` means the stored content-mean rather than a position.
POSITIONS = {
    "prompt mean (no special tokens)": ("mean", None),
    "assistant header": ("values", -5),
    "final token": ("values", -1),
}


def series(blob: Any, key: str, suffix: str, position: int | None, slot: int, column: int) -> np.ndarray:
    """One concept's value at one readout position, across the rungs of one ladder.

    :param blob: the loaded npz.
    :param key: `ladder.rendering`.
    :param suffix: `mean` or `values`.
    :param position: token index, or None for the content mean.
    :param slot: index along the layer axis.
    :param column: index along the direction axis.

    :return: `[rung]`.
    """
    if suffix == "mean":
        return blob[f"{key}.mean"][:, COSINE, slot, column]
    return blob[f"{key}.values"][:, COSINE, position, slot, column]


def zscores(blob: Any, render: str, suffix: str, position: int | None, concepts: int) -> dict[str, np.ndarray]:
    """Swing z-scores at one readout position, per ladder.

    :param blob: the loaded npz.
    :param render: `chat` or `raw`.
    :param suffix: `mean` or `values`.
    :param position: token index, or None.
    :param concepts: number of real directions.

    :return: ladder -> `[layer, column]`.
    """
    out = {}
    for ladder in LADDERS:
        key = f"{ladder}.{render}"
        block = blob[f"{key}.mean"] if suffix == "mean" else blob[f"{key}.values"][:, :, position]
        swing = block[-1, COSINE] - block[0, COSINE]
        control = swing[:, concepts:]
        out[ladder] = (swing - control.mean(-1, keepdims=True)) / np.maximum(control.std(-1, keepdims=True), 1e-12)
    return out


def page(pdf: Any, blob: Any, index: dict[str, Any], table: pd.DataFrame, render: str,
         label: str, suffix: str, position: int | None, concepts: int, controls: int,
         slot: int, layer: int, ylim: float, seed: int) -> dict[str, Any]:
    """Draw one readout position's spectrum and return its headline numbers.

    :param pdf: open `PdfPages`.
    :param blob: the loaded npz.
    :param index: manifest keyed by `ladder.rendering`.
    :param table: per-concept metadata, one row per pair.
    :param render: `chat` or `raw`.
    :param label: human name for this position.
    :param suffix: `mean` or `values`.
    :param position: token index, or None.
    :param concepts: number of real directions.
    :param controls: number of random directions.
    :param slot: index along the layer axis.
    :param layer: the block number, for titles.
    :param ylim: shared symmetric y-limit.
    :param seed: which random directions to draw.

    :return: counts and agreement statistics for this position.
    """
    z = zscores(blob, render, suffix, position, concepts)
    strength = np.abs(z["tylenol"][slot, :concepts])
    order = np.argsort(-strength)

    rows = [
        ("rank 1-6 of 1036", [0, 1, 2, 3, 4, 5]),
        ("rank 50, 100, 150, 200, 250, 300", [49, 99, 149, 199, 249, 299]),
        ("rank 400, 500, 600, 700, 850, 1000", [399, 499, 599, 699, 849, 999]),
    ]
    fig, axes = plt.subplots(4, 6, figsize=(13.5, 9.4))
    for row, (caption, ranks) in enumerate(rows):
        for col, rank in enumerate(ranks):
            pair = int(order[rank])
            ax = axes[row, col]
            for ladder, colour in zip(LADDERS, COLOURS):
                key = f"{ladder}.{render}"
                values = series(blob, key, suffix, position, slot, pair)
                ax.plot(index[key]["doses"], values - values[0], marker="o", ms=2.5, lw=1.1,
                        color=colour, label=ladder)
            ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylim(-ylim, ylim)
            ax.set_title(f"#{rank + 1}  {str(table.iloc[pair].concept)[:30]}\n"
                         f"z={z['tylenol'][slot, pair]:.1f}  steps={z['steps'][slot, pair]:.1f}",
                         fontsize=6.5)
            ax.tick_params(labelsize=5.5)
        axes[row, 0].set_ylabel(f"{caption}\n" r"$\Delta\cos(\hat v,\hat h)$ vs lowest rung", fontsize=6)

    picks = np.random.default_rng(seed).choice(controls, size=6, replace=False)
    for col, pick in enumerate(picks):
        ax = axes[3, col]
        for ladder, colour in zip(LADDERS, COLOURS):
            key = f"{ladder}.{render}"
            values = series(blob, key, suffix, position, slot, concepts + int(pick))
            ax.plot(index[key]["doses"], values - values[0], marker="o", ms=2.5, lw=1.1,
                    color=colour, label=ladder)
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(f"random direction #{pick}", fontsize=6.5)
        ax.tick_params(labelsize=5.5)
    axes[3, 0].set_ylabel("512 random directions\n" r"$\Delta\cos(\hat v,\hat h)$ vs lowest rung", fontsize=6)
    axes[0, 0].legend(fontsize=5)

    danger = int((strength >= 4).sum())
    leak = int((np.abs(z["steps"][slot, :concepts]) >= 4).sum())
    fig.suptitle(
        f"{label}  --  block {layer}, ranked by this position's own statistic.  "
        f"{danger} concepts respond to dose, {leak} to step count "
        f"(ratio {danger / max(leak, 1):.0f}x).  "
        r"y: $\Delta\cos(\hat v,\hat h)$ vs lowest rung, shared scale.", fontsize=9.5)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return {
        "label": label,
        "n_danger": danger,
        "n_steps": leak,
        "ratio": danger / max(leak, 1),
        "median_z": float(np.median(strength)),
        "max_z": float(strength.max()),
        "top": [str(table.iloc[int(p)].concept) for p in order[:8]],
        "z_tylenol": z["tylenol"][slot, :concepts],
    }


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    blob = np.load(args.readout, allow_pickle=False)
    manifest = json.loads(str(blob["manifest"]))
    meta = json.loads(str(blob["meta"]))
    index = {f"{e['ladder']}.{e['rendering']}": e for e in manifest}
    concepts, controls = meta["concepts"], meta["controls"]
    slot = LAYERS.index(args.layer)

    pairs = pd.read_parquet(args.pairs)
    if "concept" not in pairs.columns:
        pairs = pairs.assign(concept=[f"pair {i}" for i in range(len(pairs))])

    chat = index[f"tylenol.{args.rendering}"]["tokens"]
    log.info(f"content span {index[f'tylenol.{args.rendering}']['spans'][0]}, "
             f"suffix {chat[-6:]}")

    ylim = 0.0
    for suffix, position in POSITIONS.values():
        block = (blob[f"tylenol.{args.rendering}.mean"] if suffix == "mean"
                 else blob[f"tylenol.{args.rendering}.values"][:, :, position])
        ylim = max(ylim, float(np.abs(block[-1, COSINE, slot, :concepts]
                                      - block[0, COSINE, slot, :concepts]).max()))
    ylim *= 1.05
    log.info(f"shared y-limit +/-{ylim:.4f}")

    results = []
    with PdfPages(args.out) as pdf:
        for label, (suffix, position) in POSITIONS.items():
            results.append(page(pdf, blob, index, pairs, args.rendering, label, suffix, position,
                                concepts, controls, slot, args.layer, ylim, args.seed))

    print(f"{'position':<34} {'dose':>6} {'steps':>6} {'ratio':>7} {'median':>7} {'max':>7}")
    print("-" * 72)
    for row in results:
        print(f"{row['label']:<34} {row['n_danger']:>6} {row['n_steps']:>6} "
              f"{row['ratio']:>6.0f}x {row['median_z']:>7.2f} {row['max_z']:>7.2f}")
    print()
    for a in range(len(results)):
        for b in range(a + 1, len(results)):
            r = np.corrcoef(results[a]["z_tylenol"], results[b]["z_tylenol"])[0, 1]
            print(f"agreement  {results[a]['label']:<34} vs {results[b]['label']:<20} r = {r:.3f}")
    print()
    for row in results:
        print(f"{row['label']}:")
        for rank, name in enumerate(row["top"], 1):
            print(f"  {rank}. {name}")
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, default=Path("dose-readout.npz"))
    parser.add_argument("--pairs", type=Path, default=Path("pairs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("dose-positions.pdf"))
    parser.add_argument("--rendering", default="chat", choices=["chat", "raw"])
    parser.add_argument("--layer", type=int, default=25, choices=LAYERS)
    parser.add_argument("--seed", type=int, default=7)
    main(parser.parse_args())
