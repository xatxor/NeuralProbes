#! /usr/bin/env python

"""Z-score each episode's per-token concept readout against its own baseline.

For every concept at every layer, the episode's own mean and standard deviation over its tokens are
subtracted out, so a value says how far this token departs from how that concept behaved across the
rest of this trajectory. That makes spikes legible without a neutral corpus.

What it deliberately cannot do is compare episodes. A concept mildly elevated for a whole trajectory
looks identical to one that never moved, because the baseline moves with it. The raw cosines are kept
next to these files precisely so any other normalisation stays available -- once only z-scores exist,
cross-episode questions cannot be asked at all.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("zscore")

# A concept that never varies within an episode has zero spread, and dividing by it would turn
# rounding noise into enormous z-scores. Those positions are reported as exactly zero instead.
FLOOR = 1e-6


def zed(scores: np.ndarray) -> np.ndarray:
    """Z-score a `[token, layer, pair]` readout against its own per-episode baseline.

    :param scores: raw cosines as `[token, layer, pair]`.

    :return: z-scores in the same shape, float16.
    """
    values = scores.astype(np.float32)
    centre = values.mean(axis=0, keepdims=True)
    spread = values.std(axis=0, keepdims=True)
    quiet = spread < FLOOR
    result = (values - centre) / np.where(quiet, 1.0, spread)
    result[np.broadcast_to(quiet, result.shape)] = 0.0
    return result.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    done = 0
    for source in sorted(args.dir.glob("*.scores.npy")):
        target = source.with_name(source.name.replace(".scores.npy", ".z.npy"))
        if target.exists() and not args.redo:
            continue
        scores = np.load(source)
        np.save(target, zed(scores))
        done += 1
        if done % 50 == 0:
            log.info(f"{done} z-scored")
    log.info(f"wrote {done} z-score files under {args.dir}")


if __name__ == "__main__":
    main()
