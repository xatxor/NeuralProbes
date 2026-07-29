#! /usr/bin/env python

"""Rank concepts by how much they *move* within a trajectory, not how large they are.

Ranking on raw cosine is useless here: all 1036 directions share a large common component (the
vectors' own manifest records `shared_component` 0.37 at L25), so the top of that list is the same
five directions in every episode regardless of content. Z-scoring each concept against its own mean
and spread inside the episode cancels that component and leaves only what varies with the text.

This is the per-trajectory self-baseline. It answers "where in this episode does this concept
spike", which is what the paper's transcript figures show. It cannot compare absolute levels across
episodes, and a concept elevated flat across a whole episode is invisible to it by construction.
"""

import argparse
import collections
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from readout import LAYERS, find

log = logging.getLogger("rank")

# Written by the vectors' manifest: the spread of cosines for a meaningless direction.
NULL_SIGMA = 0.015625
# Roles the model itself produced, as opposed to what the environment handed it.
SPOKEN = ("thinking", "tool_call", "answer")


def zed(scores: np.ndarray) -> np.ndarray:
    """Z-score every concept against its own distribution within this trajectory.

    :param scores: `[token, layer, pair]` as saved by readout.py.

    :return: the same shape, in standard deviations from that episode's own mean.
    """
    values = scores.astype(np.float32)
    mean = values.mean(axis=0, keepdims=True)
    # The floor stops a concept that barely varies from turning rounding noise into a huge z.
    spread = np.maximum(values.std(axis=0, keepdims=True), NULL_SIGMA / 4)
    return (values - mean) / spread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("episodes/stage1"))
    parser.add_argument("--layer", type=int, default=25, choices=sorted(LAYERS))
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--detail", default=None, help="episode stem to show peak context for")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    slot = list(LAYERS).index(args.layer)

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    names = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    classes = [row["class_name"] for row in rows]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    peaks: dict[int, list[float]] = collections.defaultdict(list)
    detail = None

    for record in sorted(args.dir.glob("*.json")):
        blob = record.with_suffix(".scores.npy")
        if not blob.is_file():
            continue
        episode = json.loads(record.read_text())
        roles = np.array(episode["roles"])
        scores = np.load(blob)
        standard = zed(scores)[:, slot, :]

        spoken = np.isin(roles, SPOKEN)
        if not spoken.any():
            continue
        peak = np.abs(standard[spoken]).max(axis=0)
        for pair, value in enumerate(peak):
            peaks[pair].append(float(value))
        if args.detail and record.stem == args.detail:
            detail = (episode, roles, standard, spoken)

    ordered = sorted(peaks, key=lambda pair: -float(np.median(peaks[pair])))
    print(f"\nL{args.layer}, {len(next(iter(peaks.values())))} episodes, ranked by median peak |z| on model-produced tokens\n")
    print(f"{'median':>7} {'min':>6} {'max':>6}  concept")
    for pair in ordered[: args.top]:
        values = peaks[pair]
        print(
            f"{np.median(values):>7.2f} {min(values):>6.2f} {max(values):>6.2f}  "
            f"{names[pair][:56]:<56} [{classes[pair][:30]}]"
        )

    if detail:
        episode, roles, standard, spoken = detail
        ids = episode["ids"]
        print(f"\n\npeak context in {args.detail}:\n")
        for pair in ordered[:5]:
            column = np.where(spoken, np.abs(standard[:, pair]), -np.inf)
            at = int(np.argmax(column))
            window = tokenizer.decode(ids[max(0, at - 24) : at + 8]).replace("\n", " ")
            print(f"  {names[pair][:52]}")
            print(f"    z={standard[at, pair]:+.2f} on {roles[at]} token {at}: ...{window[-160:]}")


if __name__ == "__main__":
    main()
