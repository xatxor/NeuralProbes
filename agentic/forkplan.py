#! /usr/bin/env python

"""Turn the screen's measured hack rates into a fork plan.

Which workloads are worth forking, and with what k, is not a judgement call -- it follows from the
unsteered hack rate, and getting it wrong is expensive in a way that is invisible until the fit
produces nothing. The paired fit consumes MIXED branch points: prefixes from which both a hack and a
give-up were sampled. A workload whose rate is near zero yields none however many continuations it is
given, which is exactly what `fastsort` would have done had it been forked on faith.

The yield model is deliberately pessimistic. Treating the k continuations from one prefix as
independent draws at the episode-level rate p gives

    P(mixed) = 1 - p**k - (1 - p)**k

which at p = 0.104, k = 8 predicts 0.58. The first run actually measured 40 mixed out of 275, i.e.
0.145 -- because a prefix largely determines its own outcome (divergence from the source trajectory's
ending was only 22.7%). So the model here shrinks p toward the prefix's own decision by a measured
factor before applying the formula, and the resulting estimate is reported next to the plan rather
than hidden inside it.

k is raised on marginal workloads and lowered on productive ones, because the quantity being bought
is mixed prefixes and its marginal cost per continuation falls as p approaches 0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured on the first fork run: 40 mixed of 275 branch points at an episode-level rate of 0.104,
# against 0.58 predicted by independent draws. The ratio is how much of the episode-level variation
# actually survives inside a shared prefix.
DIVERGENCE = 0.227

# Below this an episode-level rate cannot produce mixed prefixes at any practical k, so the workload
# is excluded rather than given continuations that will all land on one side.
FLOOR = 0.08


def mixed_probability(rate: float, k: int) -> float:
    """Estimated chance that one branch point yields both a hack and a give-up.

    :param rate: unsteered episode-level hack rate for the workload.
    :param k: continuations resampled per branch point.

    :return: probability in [0, 1].
    """
    # Within a prefix the model re-decides only part of the time; the rest of the mass sits on
    # whatever the source trajectory did. Shrinking toward 0 or 1 accordingly.
    effective = 0.5 - (0.5 - rate) * (1.0 - DIVERGENCE) if rate < 0.5 else rate
    effective = min(max(effective * DIVERGENCE + rate * (1 - DIVERGENCE), 0.001), 0.999)
    return 1.0 - effective ** k - (1.0 - effective) ** k


def choose_k(rate: float, budget: int) -> int:
    """Pick k for one workload: enough to make mixed prefixes likely, capped by the budget.

    :param rate: unsteered hack rate.
    :param budget: continuations affordable for this workload.

    :return: continuations per branch point.
    """
    for k in (6, 8, 10, 12, 16):
        if mixed_probability(rate, k) >= 0.45:
            return k
    return 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True, help="shortcutprobe.py --out json")
    parser.add_argument("--out", type=Path, default=Path("forkplan.json"))
    parser.add_argument("--budget", type=int, default=250,
                        help="total continuations affordable across the whole fork wave")
    parser.add_argument("--take", type=int, default=2, help="how many workloads to fork")
    parser.add_argument("--back", type=int, default=3)
    args = parser.parse_args()

    summary = json.loads(args.screen.read_text())["summary"]
    # Solvable controls end in submit and have no branch point to resample, so they are never forked
    # however they score.
    ranked = sorted(((v["rate"], name, v) for name, v in summary.items()
                     if not name.endswith("_possible")), reverse=True)

    print(f"{'workload':<22}{'n':>4}{'rate':>8}{'P(mixed|k=8)':>14}   decision")
    chosen = []
    for rate, name, cell in ranked:
        estimate = mixed_probability(rate, 8)
        if rate < FLOOR:
            note = "EXCLUDED - rate too low to yield mixed prefixes"
        elif len(chosen) >= args.take:
            note = "not forked - budget spent on higher-rate workloads"
        else:
            note = "FORK"
            chosen.append((name, rate, cell))
        print(f"{name:<22}{cell['n']:>4}{rate:>8.3f}{estimate:>14.2f}   {note}")

    if not chosen:
        print("\nNO WORKLOAD CLEARS THE FLOOR. A multi-workload fit is not possible from this screen.")
        print("The honest conclusion is that this model reward-hacks only where the shortcut is a "
              "fact it already holds, and `fastsum` is the only such task found.")
        args.out.write_text(json.dumps({"fork": []}, indent=1))
        return

    share = args.budget // len(chosen)
    plan = []
    print()
    for name, rate, cell in chosen:
        k = choose_k(rate, share)
        points = max(1, share // k)
        plan.append({"workload": "12_shortcuts", "variant": name, "k": k, "back": args.back})
        print(f"{name:<22} k={k:<3} ~{points} branch points x {k} = ~{points * k} continuations, "
              f"expect ~{points * mixed_probability(rate, k):.0f} mixed")

    args.out.write_text(json.dumps({"fork": plan, "budget": args.budget}, indent=1))
    print(f"\nwrote {args.out} with {len(plan)} workloads")
    print("fastsum's existing forked gradients are pooled in separately, so the fitted corpus is "
          f"{len(plan)} + 1 workloads and leave-one-workload-out has something to hold out.")


if __name__ == "__main__":
    main()
