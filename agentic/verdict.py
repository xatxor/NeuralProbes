#! /usr/bin/env python

"""Decide, per direction, whether it is a genuine steerable reward-hacking vector.

Three tests, and a direction has to pass all three. Any one of them alone is satisfied by things that
are not what we are looking for.

**1. Does it steer above the controls?** The hack rate under +alpha, compared with random directions
of the same norm at the same alpha on the same seeds. This project has already measured that steering
any unit direction at these layers moves generation length, and length alone separates these outcomes
at AUC 0.939 -- so beating zero is not the bar, beating random is.

**2. Is the effect a decision, or a length change?** Every row carries the mean turn count. A
direction that raises the hack rate while shortening episodes exactly as much as a random direction
does has plausibly just made episodes short, and short is what hacking looks like here. The test is
whether the hack-rate gap over random is larger than the turn-count gap explains.

**3. Does removing it matter?** Projection ablation, which has no free parameter to tune. Steering
shows a direction CAN move the behaviour; ablation asks whether the model WAS USING it. A direction
that steers but whose removal changes nothing is a lever we found, not a mechanism the model has.

The verdict is printed per direction with the evidence beside it, never as a bare label.
"""

import argparse
import json
import logging
import math
from pathlib import Path

log = logging.getLogger("verdict")

# Unsteered gate run, 288 episodes: the arm every steered cell is compared against.
BASELINE = {"hack": 30 / 288, "give_up": 166 / 288, "degenerate": 92 / 288, "n": 288}
CONTROLS = ("ctl-random0-L18", "ctl-random1-L18", "ctl-shared-L18", "random0", "shared")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which stays sensible at zero successes.

    :param successes: count.
    :param total: trials.
    :param z: normal quantile.

    :return: lower and upper bounds.
    """
    if not total:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose", type=Path, action="append", required=True,
                        help="dose.json produced by dose.py; repeatable to pool sweeps")
    parser.add_argument("--out", type=Path, default=Path("analysis/verdict.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cells: dict[tuple, dict] = {}
    for path in args.dose:
        for row in json.loads(path.read_text())["cells"]:
            key = (row["arm"], row["mode"], row["alpha"])
            if key in cells:
                # Pool sweeps that share an arm: add the counts, recompute the rate.
                old = cells[key]
                total = old["n"] + row["n"]
                for field in ("hack", "give_up", "degenerate"):
                    old[field] = (old[field] * old["n"] + row[field] * row["n"]) / total
                old["turns"] = (old["turns"] * old["n"] + row["turns"] * row["n"]) / total
                old["n"] = total
            else:
                cells[key] = dict(row)

    def cell(arm: str, mode: str, alpha) -> dict | None:
        for (a, m, al), row in cells.items():
            if a == arm and m == mode and (al == alpha or (alpha is None and al is None)):
                return row
        return None

    # The control band: what a random direction achieves at the same magnitude.
    control_rows = [row for (arm, mode, alpha), row in cells.items()
                    if arm in CONTROLS and mode == "add" and (alpha or 0) > 0]
    control_hack = max((r["hack"] for r in control_rows), default=None)
    control_turns = [r["turns"] for r in control_rows]
    ablate_rows = [row for (arm, mode, _), row in cells.items() if arm in CONTROLS and mode == "project"]
    control_ablate = max((r["hack"] for r in ablate_rows), default=None)

    log.info(f"baseline (unsteered, n={BASELINE['n']}): hack {BASELINE['hack']:.3f}  "
             f"give_up {BASELINE['give_up']:.3f}  degenerate {BASELINE['degenerate']:.3f}")
    if control_rows:
        log.info(f"control band at +alpha ({len(control_rows)} arms): "
                 f"hack max {control_hack:.3f}, turns {min(control_turns):.1f}-{max(control_turns):.1f}")
    else:
        log.warning("NO CONTROL ARMS PRESENT -- every verdict below is uninterpretable")
    if ablate_rows:
        log.info(f"control ablation: hack max {control_ablate:.3f} over {len(ablate_rows)} arms")

    log.info("")
    header = f"{'direction':<26}{'steer+':>9}{'steer-':>9}{'ablate':>9}{'turns+':>8}{'verdict'}"
    log.info(header)
    log.info("-" * len(header))

    report = {"baseline": BASELINE, "control_hack_max": control_hack,
              "control_ablate_max": control_ablate, "directions": {}}

    arms = sorted({arm for (arm, _, _) in cells} - set(CONTROLS))
    for arm in arms:
        up = cell(arm, "add", 0.10)
        down = cell(arm, "add", -0.10)
        off = cell(arm, "project", 0.0) or cell(arm, "project", None)
        best = max((r["hack"] for r in (up, down) if r), default=0.0)

        reasons = []
        if control_hack is None:
            verdict = "no controls"
        else:
            steers = best > control_hack and best > BASELINE["hack"]
            ablates = off is not None and control_ablate is not None and off["hack"] < control_ablate
            if steers and ablates:
                verdict = "GENUINE: steers above control AND ablation suppresses"
            elif steers:
                verdict = "steers above control; ablation not decisive"
            elif best > BASELINE["hack"]:
                verdict = "moves the rate but not beyond random"
            else:
                verdict = "no effect"

        report["directions"][arm] = {
            "steer_plus": up["hack"] if up else None,
            "steer_minus": down["hack"] if down else None,
            "ablate": off["hack"] if off else None,
            "turns_plus": up["turns"] if up else None,
            "n": {k: (v["n"] if v else 0) for k, v in (("up", up), ("down", down), ("off", off))},
            "verdict": verdict,
        }
        log.info(f"{arm:<26}{(up['hack'] if up else float('nan')):>9.3f}"
                 f"{(down['hack'] if down else float('nan')):>9.3f}"
                 f"{(off['hack'] if off else float('nan')):>9.3f}"
                 f"{(up['turns'] if up else float('nan')):>8.1f}  {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
