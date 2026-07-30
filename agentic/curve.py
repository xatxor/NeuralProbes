#! /usr/bin/env python

"""Plot how far into an episode the model has to be before its state predicts the ending.

Two panels. The left one grows a prefix -- turn 0 alone, then turns 0-1, then 0-2 and so on -- so it
answers "having watched k turns, can the ending be called". The right one takes single turns in
isolation, which answers the different question of whether any one turn carries what the prefix
carries or whether the prefix only works by accumulating tokens.

Only raw cosines are plotted. A per-episode z-score is standardised against that episode's own mean
and standard deviation over every token, final turn included, so a z-scored prefix has been scaled by
its own future and cannot support a claim about prediction however well it scores.

Every point requires the episode to have actually reached that turn, so the episodes behind a point
have all been observed for the same number of turns. Without that, short episodes -- which is what
reward hacks are -- would contribute their whole trajectory while long ones contributed a fragment.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("curve")

PAGE = (12.5, 5.6)
INK = "#1b1b1b"
GREY = "#8d8d8d"
SERIES = {
    "hack_vs_rest": ("hacked vs everything else", "#d1495b"),
    "hack_vs_giveup": ("hacked vs gave up", "#2e86ab"),
}
PREFIXES = (1, 2, 3, 4, 6, 8)
SINGLES = (0, 1, 2, 3)


def gather(report: dict, keys: list[str], comparison: str) -> dict[str, list]:
    """Pull the cross-validated score and its null for a sequence of feature sets.

    :param report: `predict.py` output.
    :param keys: feature-set keys in plotting order.
    :param comparison: which class contrast.

    :return: parallel lists of position, score, null, p and episode count.
    """
    out: dict[str, list] = {"x": [], "auc": [], "null": [], "p": [], "n": []}
    for position, key in enumerate(keys):
        cv = report["multivariate"].get(key, {}).get(comparison)
        block = report["univariate"].get(key, {}).get(comparison)
        if not cv or not block:
            continue
        out["x"].append(position)
        out["auc"].append(cv["auc"])
        out["null"].append(cv["null_p95"])
        out["p"].append(cv["p"])
        out["n"].append(block["n_pos"])
    return out


def panel(axis, report: dict, keys: list[str], ticks: list[str], heading: str) -> None:
    """Draw one predictability panel."""
    for comparison, (label, colour) in SERIES.items():
        got = gather(report, keys, comparison)
        if not got["x"]:
            continue
        axis.plot(got["x"], got["auc"], color=colour, linewidth=2.0, marker="o", markersize=5, label=label)
        # The 95th percentile of the shuffled-label score is the line a point has to clear; it moves
        # with n, so it is drawn per point rather than as one horizontal rule.
        axis.plot(got["x"], got["null"], color=colour, linewidth=0.9, linestyle=":", alpha=0.7)
        for x, y, p, n in zip(got["x"], got["auc"], got["p"], got["n"]):
            if p < 0.05:
                axis.plot([x], [y], marker="o", markersize=9, markerfacecolor="none",
                          markeredgecolor=colour, markeredgewidth=1.4)
            axis.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, -14),
                          ha="center", fontsize=6.5, color=GREY)

    shape = report["baseline"].get("shape_model", {}).get("hack_vs_rest", {}).get("auc")
    if shape:
        axis.axhline(shape, color="#8d6e63", linewidth=1.3, linestyle="--")
        axis.annotate(f"episode shape alone, {shape:.3f}", (0.02, shape - 0.035),
                      xycoords=("axes fraction", "data"), fontsize=7.5, color="#8d6e63")
    axis.axhline(0.5, color=INK, linewidth=0.8, linestyle=":")
    axis.set_xticks(range(len(ticks)))
    axis.set_xticklabels(ticks, fontsize=8.5)
    # Wide enough to hold a point that lands below chance rather than clipping it off the axis --
    # a vanishing line reads as missing data when it is actually a measurement.
    axis.set_ylim(0.25, 1.0)
    axis.set_ylabel("leave-one-out AUC", fontsize=9)
    axis.set_title(heading, fontsize=10, color=INK)
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.tick_params(labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("curve/predict.json"))
    parser.add_argument("--out", type=Path, default=Path("curve/curve.pdf"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = json.loads(args.source.read_text())

    figure, axes = plt.subplots(1, 2, figsize=PAGE, sharey=True)
    panel(axes[0], report,
          [f"cos:upto{k}|ctl" for k in PREFIXES],
          [f"{k}" for k in PREFIXES],
          "turns 0..k observed, episode shape removed")
    axes[0].set_xlabel("turns of reasoning observed", fontsize=9)
    panel(axes[1], report,
          [f"cos:turn{k}|ctl" for k in SINGLES],
          [f"turn {k}" for k in SINGLES],
          "one turn in isolation, episode shape removed")
    axes[1].set_xlabel("which turn", fontsize=9)
    axes[0].legend(fontsize=8.5, loc="upper left", frameon=False)

    figure.suptitle(
        "How early can the ending be called?  ·  raw cosines only  ·  ring = p < 0.05 against shuffled labels",
        fontsize=11.5)
    figure.text(0.5, 0.015,
                "dotted line = 95th percentile of the shuffled-label score at that n  ·  small grey number = hacked episodes contributing",
                fontsize=7.5, color=GREY, ha="center")
    figure.tight_layout(rect=(0, 0.045, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
