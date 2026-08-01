#! /usr/bin/env python

"""Declare the steering arms for the agentic evaluation.

Written as a file rather than inline in the shell so that the comparison is explicit and auditable:
every direction that gets a GPU-hour is listed here with the reason it is in the sweep, and the
controls are declared beside the candidates instead of being remembered later.

The head-to-head this exists for: for the SAME concept -- `shortcut acceptance || shortcut
rejection`, which is as close to a definition of reward hacking as the ontology contains -- steer with
the published activation-space vector and with the gradient-space one extracted here, at identical
alphas on identical seeds. Whichever moves the hack rate more, moves it in the right direction, and
moves it further than a random direction of the same norm, is the better vector. That is a
measurement rather than an argument, which is the point.

Two controls are mandatory and neither is optional decoration:

- `random0` -- a seeded random unit direction. This project has already measured that steering ANY
  unit direction at these layers changes generation length, and length alone separates the outcomes at
  AUC 0.939. An arm that beats nothing beats random.
- `shared` -- the normalised mean of all 1036 published directions. If an effect reproduces here, it
  belongs to the vector set's shared component rather than to any concept in it.

alpha = 0 is deliberately absent: 288 unsteered gate episodes already exist and are the baseline.
"""

import argparse
import json
from pathlib import Path

# Alphas as fractions of the measured residual norm at the steering layer. Symmetric, because a
# direction that only works in one sign is a plausible artifact of pushing the model off-distribution
# rather than a handle on the behaviour. Two magnitudes rather than four: at 300s per episode the
# four-point dose curve costs six GPU-hours more than the slot allows, and a sign test with a
# matched random control answers "does it steer" -- the shape of the curve is a later luxury.
ALPHAS = (0.10, -0.10)

# `pair` reads a published concept vector by index; `file` reads an extracted .npy; the bare names
# are agent.py's built-in controls.
ARMS = [
    # --- outcome-supervised directions, goal 1 ---
    {"name": "trained_thinking", "direction": "file:vectors/trained-L18-r50.npy",
     "why": "true DPO objective, deliberation window; reproducible across hyperparameters (cos .925)"},
    {"name": "trained_all", "direction": "file:vectors/trained-L18-r15.npy",
     "why": "true DPO, all emitted tokens; scores higher but reads the tool call too"},
    {"name": "grpo_hack_giveup", "direction": "file:vectors/grpo-L18-hack_vs_giveup-mean.npy",
     "why": "one-step GRPO; drove the hack rate to 0.944 in the first sweep"},
    {"name": "dpo_linearised", "direction": "file:vectors/vector-L18-mean.npy",
     "why": "linearised DPO, the first sweep's other mover"},
    # --- goal 2 head-to-head on one concept ---
    {"name": "story_grad_872", "direction": "file:vectors/gradvec-872-L18.npy",
     "why": "gradient-space 'shortcut acceptance || rejection'"},
    {"name": "story_pub_872", "direction": "file:vectors/pub872-L18.npy",
     "why": "PUBLISHED activation-space vector, identical concept"},
    # --- controls. These were silently absent from sweep 1: `random0`/`shared`/`872` resolve through
    # find('diff.safetensors'), and the job container caches the model but NOT the vectors repo, so
    # every one of those arms failed. They are files now.
    {"name": "ctl_random0", "direction": "file:vectors/ctl-random0-L18.npy",
     "why": "matched-norm random direction: steering ANY direction moves generation length"},
    {"name": "ctl_random1", "direction": "file:vectors/ctl-random1-L18.npy",
     "why": "a second random draw, so the control has its own spread"},
    {"name": "ctl_shared", "direction": "file:vectors/ctl-shared-L18.npy",
     "why": "normalised mean of all 1036 published directions"},
]

# Ablation is separate: it has no alpha, and it asks the harder question -- whether the model was
# USING the direction, not merely whether the direction can push it.
ABLATE = ["trained_thinking", "grpo_hack_giveup", "story_grad_872", "ctl_random0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("arms.json"))
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-from", type=int, default=1000)
    args = parser.parse_args()

    # Emitted in priority order, and the shard runner walks the list in order under a wall-clock
    # budget. So if the budget bites, what is lost is the tail -- never the head-to-head or the
    # control that makes it interpretable.
    jobs = []
    for alpha in ALPHAS:
        # Round one: every candidate against its control at one magnitude. This alone answers the
        # question if nothing else finishes.
        for arm in ARMS:
            jobs.append({"name": arm["name"], "direction": arm["direction"],
                         "mode": "add", "alpha": alpha, "priority": 0 if alpha > 0 else 1})
    for arm in ARMS:
        if arm["name"] in ABLATE:
            jobs.append({"name": arm["name"], "direction": arm["direction"],
                         "mode": "project", "alpha": 0.0, "priority": 2})

    plan = {
        "seeds": args.seeds,
        "seed_from": args.seed_from,
        "arms": ARMS,
        "jobs": jobs,
        "episodes": len(jobs) * args.seeds,
    }
    args.out.write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} arm-configurations x {args.seeds} seeds = {plan['episodes']} episodes")
    for job in jobs:
        print(f"  {job['name']:<20} {job['mode']:<8} alpha={job['alpha']:+.2f}  {job['direction']}")


if __name__ == "__main__":
    main()
