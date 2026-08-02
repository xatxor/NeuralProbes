#! /usr/bin/env python

"""Read the steered-against-unsteered readout distributions.

Each concept is standardised by its own BASELINE mean and standard deviation across the prompt set,
so the unsteered distribution is N(0,1) by construction and a steered arm's displacement is read
directly in standard deviations. Separation is reported as Cohen's d and as the AUC of the two
distributions, which is the probability that a randomly drawn steered prompt reads higher than a
randomly drawn unsteered one -- 0.5 is complete overlap, 1.0 is complete separation.

Three numbers matter and they answer different questions.

  ON-TARGET, AT THE INJECTION BLOCK   How far the steered concept moves where the vector was added.
      This is close to arithmetic and is reported only so it can be discounted.
  ON-TARGET, AT A LATER BLOCK         Whether the perturbation survives further processing. This is
      the honest "did it take" number.
  OFF-TARGET COUNT                    How many of the other 1035 concepts also separate. This is
      selectivity, and it is the number the experiment exists for. A direction that moves one concept
      and leaves the rest alone is a concept vector; one that moves three hundred is a perturbation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def separation(baseline: np.ndarray, steered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cohen's d and AUC per concept.

    :param baseline: `[prompt, concept]` unsteered readouts.
    :param steered: `[prompt, concept]` steered readouts.

    :return: `(d, auc)`, each `[concept]`.
    """
    mean_b, mean_s = baseline.mean(0), steered.mean(0)
    var_b, var_s = baseline.var(0, ddof=1), steered.var(0, ddof=1)
    pooled = np.sqrt((var_b + var_s) / 2.0) + 1e-9
    d = (mean_s - mean_b) / pooled
    # Rank-based AUC, so it makes no normality assumption even though the picture is drawn as bells.
    n = baseline.shape[0]
    auc = np.empty(baseline.shape[1], dtype=np.float64)
    for c in range(baseline.shape[1]):
        joint = np.concatenate([baseline[:, c], steered[:, c]])
        ranks = joint.argsort().argsort().astype(np.float64) + 1.0
        auc[c] = (ranks[n:].sum() - n * (n + 1) / 2.0) / (n * n)
    return d, auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("analysis/zdist.npz"))
    parser.add_argument("--pairs", type=Path, default=Path("probes-notemplate/pairs.parquet"))
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="|d| above which a concept counts as moved")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = np.load(args.data, allow_pickle=True)
    table, arms = payload["table"], [str(a) for a in payload["arms"]]
    layers = [int(x) for x in payload["layers"]]
    inject = int(payload["inject_layer"])

    names = {}
    try:
        import pyarrow.parquet as pq
        frame = pq.read_table(args.pairs).to_pydict()
        for index, (concept, antagonist) in enumerate(zip(frame["concept"], frame["antagonist"])):
            names[index] = f"{concept} || {antagonist}"
    except Exception:                                              # noqa: BLE001 - labels are cosmetic
        pass

    base = arms.index("baseline")
    report: dict = {"inject_layer": inject, "layers": layers, "threshold": args.threshold, "arms": {}}

    for l, layer in enumerate(layers):
        raw_base = table[base, :, l, :]
        centre, spread = raw_base.mean(0), raw_base.std(0, ddof=1) + 1e-9
        print(f"\n{'=' * 96}\nREAD AT BLOCK {layer}"
              + ("   (the injection block -- displacement here is close to arithmetic)"
                 if layer == inject else "   (downstream of the injection)")
              + f"\n{'=' * 96}")
        print(f"{'arm':<18}{'target d':>10}{'target AUC':>12}{'off |d|>' + str(args.threshold):>14}"
              f"{'median off |d|':>16}{'max off':>10}  worst off-target")
        for a, arm in enumerate(arms):
            if arm == "baseline":
                continue
            z_base = (raw_base - centre) / spread
            z_arm = (table[a, :, l, :] - centre) / spread
            d, auc = separation(z_base, z_arm)

            target = None
            if arm.startswith("pair"):
                target = int(arm[4:].split("@")[0])
            mask = np.ones(d.shape, dtype=bool)
            if target is not None:
                mask[target] = False
            off = np.abs(d[mask])
            moved = int((off > args.threshold).sum())
            worst = int(np.argsort(-off)[0])
            worst_index = np.arange(d.shape[0])[mask][worst]
            head = (f"{d[target]:>10.2f}{auc[target]:>12.3f}" if target is not None
                    else f"{'--':>10}{'--':>12}")
            print(f"{arm:<18}{head}{moved:>14}{np.median(off):>16.2f}{off.max():>10.2f}  "
                  f"{names.get(worst_index, worst_index)[:40]}")
            report["arms"].setdefault(arm, {})[str(layer)] = {
                "target_d": float(d[target]) if target is not None else None,
                "target_auc": float(auc[target]) if target is not None else None,
                "off_target_moved": moved,
                "off_target_median_abs_d": float(np.median(off)),
                "off_target_max_abs_d": float(off.max()),
                "off_target_fraction": moved / float(mask.sum()),
            }

    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
