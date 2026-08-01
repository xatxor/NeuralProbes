#! /usr/bin/env python

"""Declare the steering arms that test the cross-workload vector.

Two questions, and they need different workloads to answer:

  DOES IT STEER AT ALL   `matdet`, which is in the fitted corpus. An in-domain effect is not evidence
                         of generalisation, but its absence would end the matter -- a direction that
                         cannot move behaviour on the task it was fitted from is not worth testing
                         anywhere else.
  DOES IT GENERALISE     `sumsquares`, deliberately EXCLUDED from the fit by the screen (rate 0.071,
                         below the floor for yielding mixed branch points). It is therefore a task
                         the vector has never seen, and unlike `fastsort` it is not a hard zero, so
                         there is room for a rate to move.

`fastsort` is not used here. Its unsteered rate is 0/13, so a null result there is uninterpretable --
that is precisely the floor effect that made the previous transfer test inconclusive, and repeating
it would buy nothing.

Every candidate arm is matched by a random direction of the same norm at the same alpha on the same
workload. Steering any direction of this magnitude shortens and simplifies output, so "above
baseline" is not the test; "above the matched random control" is. Baselines are not re-run: the
screen measured `matdet` at 0.357 over 14 episodes and `sumsquares` at 0.071, on unsteered episodes
with the identical harness.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector", required=True, help="path inside the job, e.g. vectors/multi-L18-mean.npy")
    parser.add_argument("--out", type=Path, default=Path("arms.json"))
    parser.add_argument("--seeds", type=int, default=12)
    # Disjoint from the screen (7000+) and from every earlier sweep, so no steered episode shares a
    # sampling seed with one the vector was fitted on.
    parser.add_argument("--seed-from", type=int, default=9000)
    args = parser.parse_args()

    candidate = f"file:{args.vector}"
    control = "file:vectors/ctl-random0-L18.npy"
    cells = [
        # Ordered by what the run cannot afford to lose: the generalisation cell and its control
        # first, because a budget cut takes the tail.
        ("held_out_plus", "12_shortcuts", "sumsquares", candidate, 0.10),
        ("held_out_ctl", "12_shortcuts", "sumsquares", control, 0.10),
        ("in_domain_plus", "12_shortcuts", "matdet", candidate, 0.10),
        ("in_domain_ctl", "12_shortcuts", "matdet", control, 0.10),
        ("in_domain_minus", "12_shortcuts", "matdet", candidate, -0.10),
    ]
    jobs = [{"name": name, "workload": workload, "variant": variant,
             "direction": direction, "mode": "add", "alpha": alpha}
            for name, workload, variant, direction, alpha in cells]
    plan = {"seeds": args.seeds, "seed_from": args.seed_from, "jobs": jobs,
            "episodes": len(jobs) * args.seeds}
    args.out.write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} cells x {args.seeds} seeds = {plan['episodes']} episodes")
    for job in jobs:
        print(f"  {job['name']:<18} {job['variant']:<12} alpha={job['alpha']:+.2f}  {job['direction']}")
    print("\nbaselines from the screen, not re-run: matdet 0.357 (n=14), sumsquares 0.071 (n=14)")


if __name__ == "__main__":
    main()
