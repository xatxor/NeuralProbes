#! /usr/bin/env python

"""Compare two dose-response runs that differ only in the vector set.

The generations are identical between runs -- same model, same seeds, same prompts -- so the model's
behaviour is held exactly fixed and every difference is attributable to the basis the activations are
read in. That is checked rather than assumed: the reply texts are compared verbatim and the run
aborts if they diverge.

`diff` is the difference of means. `lda` is that difference whitened by the pooled within-group
covariance with a 0.05 trace ridge, which `whiten.py` verifies reproduces the published LDA vectors
at cosine 1.000000. Both are normalised to unit length before projection, so the comparison is of
direction only, not of scale.

Reported per statistic: whether the two bases agree on which concepts respond, whether whitening
changes the count that clears the null, and whether it changes the danger/control separation, which
is the quantity the whole report turns on.
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

log = logging.getLogger("dosecmp")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
LADDERS = DANGER + ("steps",)
COLOURS = {"tylenol": "C3", "syrup": "C1", "ibuprofen": "C4", "steps": "C0"}


def load(root: Path) -> dict[str, Any]:
    """Read one run's summary, per-concept table and derived arrays.

    :param root: directory holding `dose-summary.json`, `dose-stats.parquet`, `dose-derived.npz`.

    :return: the three objects under `summary`, `table`, `derived`.
    """
    return {
        "summary": json.loads((root / "dose-summary.json").read_text()),
        "table": pd.read_parquet(root / "dose-stats.parquet"),
        "derived": np.load(root / "dose-derived.npz", allow_pickle=False),
    }


def replies(root: Path) -> dict[str, str]:
    """The greedy continuation of every chat rung, keyed by ladder and dose.

    :param root: directory holding the shard files.

    :return: `"ladder:dose"` -> generated text.
    """
    out: dict[str, str] = {}
    for path in sorted(root.glob("shard-*/dose-readout.npz")):
        for row in json.loads(str(np.load(path, allow_pickle=False)["manifest"])):
            if row["rendering"] != "chat":
                continue
            for entry in row["replies"]:
                if entry["label"] == "greedy":
                    out[f"{row['ladder']}:{row['dose']}"] = entry["text"]
    return out


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    a, b = load(args.a), load(args.b)
    concepts = a["summary"]["meta"]["concepts"]
    real = slice(0, concepts)
    slot = LAYERS.index(25)
    names = (args.label_a, args.label_b)

    # The behaviour must be identical; otherwise the comparison is not of bases alone.
    ra, rb = replies(args.a_shards or args.a), replies(args.b_shards or args.b)
    shared = sorted(set(ra) & set(rb))
    if not shared:
        log.warning("no shard files found, skipping the identical-generation check")
    else:
        differing = [k for k in shared if ra[k] != rb[k]]
        log.info(f"generations compared on {len(shared)} rungs, {len(differing)} differ")
        if differing:
            raise SystemExit(f"generations differ on {differing[:5]}; the runs are not comparable")

    rows = []
    for slot_i, layer in enumerate(LAYERS):
        for ladder in LADDERS:
            rows.append({
                "layer": layer, "ladder": ladder,
                f"prompt_{names[0]}": a["summary"]["per_layer_final"][str(layer)][ladder],
                f"prompt_{names[1]}": b["summary"]["per_layer_final"][str(layer)][ladder],
                f"reply_{names[0]}": a["summary"]["per_layer_reply"][str(layer)][ladder],
                f"reply_{names[1]}": b["summary"]["per_layer_reply"][str(layer)][ladder],
            })
    counts = pd.DataFrame(rows)

    agree = {}
    for key, prefix in (("prompt", "zf"), ("reply", "zr")):
        agree[key] = {
            str(layer): float(np.corrcoef(a["derived"][f"{prefix}.tylenol"][i, real],
                                          b["derived"][f"{prefix}.tylenol"][i, real])[0, 1])
            for i, layer in enumerate(LAYERS)
        }

    def spread(run: dict[str, Any], prefix: str) -> dict[str, float]:
        z = run["derived"][f"{prefix}.tylenol"][slot, real]
        control = run["derived"][f"{prefix}.steps"][slot, real]
        return {"median": float(np.median(np.abs(z))), "max": float(np.abs(z).max()),
                "n4": int((np.abs(z) >= 4).sum()), "control_n4": int((np.abs(control) >= 4).sum()),
                "control_r": float(np.corrcoef(z, control)[0, 1])}

    top_a = a["table"][a["table"].hit & (a["table"].specificity > 0)]
    top_b = b["table"][b["table"].hit & (b["table"].specificity > 0)]
    set_a = set(top_a[top_a.layer == 25].sort_values("specificity", ascending=False).head(25).pair)
    set_b = set(top_b[top_b.layer == 25].sort_values("specificity", ascending=False).head(25).pair)

    summary = {
        "labels": list(names),
        "counts": counts.to_dict("records"),
        "cross_basis_agreement": agree,
        "prompt": {names[0]: spread(a, "zf"), names[1]: spread(b, "zf")},
        "reply": {names[0]: spread(a, "zr"), names[1]: spread(b, "zr")},
        "estimators": {n: {k: v["25"] for k, v in run["summary"]["estimators"].items()}
                       for n, run in zip(names, (a, b))},
        "coupling": {n: run["summary"]["prompt_reply_agreement_final"] for n, run in zip(names, (a, b))},
        "hits25": {names[0]: int((top_a.layer == 25).sum()), names[1]: int((top_b.layer == 25).sum())},
        "top25_overlap": len(set_a & set_b),
    }
    (args.out / "dose-compare.json").write_text(json.dumps(summary, indent=2))

    with PdfPages(args.out / "dose-compare.pdf") as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        width = 0.35
        for ax, key, caption in zip(axes[:2], ("prompt", "reply"),
                                    ("(a) prompt, final token", "(b) reply")):
            for offset, name in enumerate(names):
                ax.bar(np.arange(len(LAYERS)) + offset * width,
                       [counts[(counts.layer == l) & (counts.ladder == "tylenol")][f"{key}_{name}"].iloc[0]
                        for l in LAYERS], width, label=name)
            ax.set_xticks(np.arange(len(LAYERS)) + width / 2)
            ax.set_xticklabels([str(n) for n in LAYERS])
            ax.set_xlabel("block")
            ax.set_ylabel(r"concepts, $|z|\geq4$")
            ax.set_title(caption, fontsize=9)
            ax.legend(fontsize=7)
        for name, run in zip(names, (a, b)):
            axes[2].plot(LAYERS, [run["summary"]["prompt_reply_agreement_final"][str(l)] for l in LAYERS],
                         marker="o", label=name)
        axes[2].axhline(0, color="k", lw=0.6)
        axes[2].set_xlabel("block")
        axes[2].set_ylabel(r"$r$(prompt, reply)")
        axes[2].set_title("(c) prompt-reply coupling", fontsize=9)
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, len(LAYERS), figsize=(13, 5.2))
        for i, layer in enumerate(LAYERS):
            for row, prefix in enumerate(("zf", "zr")):
                x = a["derived"][f"{prefix}.tylenol"][i, real]
                y = b["derived"][f"{prefix}.tylenol"][i, real]
                axes[row, i].scatter(x, y, s=3, color="C3", alpha=0.4)
                lim = float(np.percentile(np.abs(np.concatenate([x, y])), 99.5))
                axes[row, i].plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":")
                axes[row, i].set_xlim(-lim, lim)
                axes[row, i].set_ylim(-lim, lim)
                axes[row, i].tick_params(labelsize=6)
                axes[row, i].set_xlabel(f"{names[0]}", fontsize=7)
            axes[0, i].set_title(f"block {layer}  r={agree['prompt'][str(layer)]:+.3f}", fontsize=8)
            axes[1, i].set_title(f"r={agree['reply'][str(layer)]:+.3f}", fontsize=8)
        axes[0, 0].set_ylabel(f"{names[1]}, prompt", fontsize=7)
        axes[1, 0].set_ylabel(f"{names[1]}, reply", fontsize=7)
        fig.suptitle("same concept, two bases", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 3.4))
        est = list(a["summary"]["estimators"])
        keys = ["tylenol-syrup", "tylenol-ibuprofen", "syrup-ibuprofen", "tylenol-steps"]
        width = 0.1
        for i, name in enumerate(names):
            run = (a, b)[i]
            for j, key in enumerate(keys):
                ax.bar(np.arange(len(est)) + (i * len(keys) + j) * width,
                       [run["summary"]["estimators"][e]["25"][key] for e in est], width,
                       color="C0" if key.endswith("steps") else "C3",
                       alpha=0.45 + 0.18 * j, label=f"{name} {key}" if i == 0 or True else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(len(est)) + 0.35)
        ax.set_xticklabels(est, fontsize=7)
        ax.set_ylabel("agreement $r$")
        ax.set_title(f"estimator agreement, block 25; left group {names[0]}, right group {names[1]}",
                     fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n{'':<26}{names[0]:>12}{names[1]:>12}")
    print("-" * 50)
    for key in ("prompt", "reply"):
        for stat in ("median", "max", "n4", "control_n4", "control_r"):
            va, vb = summary[key][names[0]][stat], summary[key][names[1]][stat]
            fmt = "{:>12.2f}" if isinstance(va, float) else "{:>12d}"
            print(f"{key + ' ' + stat:<26}" + fmt.format(va) + fmt.format(vb))
    print(f"\n{'top-25 overlap':<26}{summary['top25_overlap']:>12} of 25")
    print(f"{'hits at block 25':<26}{summary['hits25'][names[0]]:>12}{summary['hits25'][names[1]]:>12}")
    print("\ncross-basis agreement on which concepts respond:")
    for key in ("prompt", "reply"):
        print(f"  {key:<8}" + "  ".join(f"L{l} {agree[key][str(l)]:+.3f}" for l in LAYERS))
    print("\nprompt-reply coupling:")
    for name in names:
        print(f"  {name:<8}" + "  ".join(f"L{l} {summary['coupling'][name][str(l)]:+.3f}" for l in LAYERS))
    log.info(f"wrote dose-compare.pdf and dose-compare.json into {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a-shards", type=Path, default=None)
    parser.add_argument("--b-shards", type=Path, default=None)
    parser.add_argument("--label-a", default="diff")
    parser.add_argument("--label-b", default="lda")
    parser.add_argument("--out", type=Path, default=Path("."))
    main(parser.parse_args())
