#! /usr/bin/env python

"""Survey which concepts spike where, across a whole directory of scored episodes.

Two things are computed for every one of the 1036 concepts at each layer:

* how hard it spikes -- the mean over episodes of its peak |z|, and how often that peak clears a
  threshold corrected for the fact that each token carries 2072 concept-layer values;
* **whether its spikes are structural**. The previous readout was topped by `uses first-person ||
  uses third-person`, which fires at turn boundaries: an artifact of the transcript's shape, not
  something the model was thinking. So the distance from each peak to the nearest turn start is
  recorded, and a concept whose peaks sit on boundaries is flagged rather than reported as a finding.

Also split by outcome, so a concept that behaves differently in episodes that reward-hacked can be
told from one that behaves the same everywhere.
"""

import argparse
import collections
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from readout import find

log = logging.getLogger("survey")

LAYERS = (18, 25)
# Each token carries 2 x 1036 values, so "some concept exceeded 3" happens at essentially every token
# by chance. Bonferroni over 2072 comparisons at 0.05 puts the bar here instead.
THRESHOLD = 4.5
# A peak this close to the start of a turn is treated as possibly structural: turn starts are where
# the transcript changes speaker, and several concepts track that rather than any content.
BOUNDARY = 8
# Roles the model itself produced. Everything else -- the system prompt, the instruction, the pytest
# output -- is text it read, not text it generated, and the prompt is identical in every episode.
GENERATED = ("thinking", "tool_call", "answer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    names = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    classes = [row["class_name"] for row in rows]

    peaks: dict[str, list] = collections.defaultdict(list)
    episodes = sorted(args.dir.glob("*.z.npy"))
    log.info(f"{len(episodes)} scored episodes in {args.dir}")

    per_concept = {
        layer: {
            "peak": np.zeros((len(episodes), len(names)), dtype=np.float32),
            "atboundary": np.zeros((len(episodes), len(names)), dtype=bool),
        }
        for layer in LAYERS
    }
    labels = []

    for index, path in enumerate(episodes):
        record = Path(str(path).replace(".z.npy", ".json"))
        episode = json.loads(record.read_text())
        starts = np.array([turn["start"] for turn in episode["turns"] if "start" in turn] or [0])
        z = np.abs(np.load(path).astype(np.float32))

        # Only the model's own tokens. The system prompt is byte-identical across all 288 episodes,
        # so a concept that likes some token in it peaks at the SAME index in every episode -- two of
        # the top three peaks for pair 857 were the same comma at token 52. Tool results are excluded
        # for the same reason: pytest output is not the model thinking.
        roles = np.array(episode["roles"][: z.shape[0]])
        mine = np.isin(roles, GENERATED)
        if not mine.any():
            continue
        positions = np.flatnonzero(mine)
        z = z[mine]

        for slot, layer in enumerate(LAYERS):
            plane = z[:, slot, :]
            where = positions[plane.argmax(axis=0)]
            per_concept[layer]["peak"][index] = plane.max(axis=0)
            # Distance from each concept's peak to the nearest turn start.
            gap = np.abs(where[:, None] - starts[None, :]).min(axis=1)
            per_concept[layer]["atboundary"][index] = gap <= BOUNDARY

        labels.append(
            {
                "episode": record.stem,
                "ending": episode.get("ending"),
                "hacked": episode.get("ending") == "submit",
                "tokens": len(episode.get("ids", [])),
            }
        )
        if (index + 1) % 50 == 0:
            log.info(f"{index + 1}/{len(episodes)}")

    hacked = np.array([entry["hacked"] for entry in labels])
    out = {"episodes": len(episodes), "hacked": int(hacked.sum()), "threshold": THRESHOLD, "concepts": []}

    for pair in range(len(names)):
        entry = {"pair": pair, "concept": names[pair], "class": classes[pair], "layers": {}}
        for layer in LAYERS:
            peak = per_concept[layer]["peak"][:, pair]
            boundary = per_concept[layer]["atboundary"][:, pair]
            entry["layers"][f"L{layer}"] = {
                "mean_peak": round(float(peak.mean()), 3),
                "max_peak": round(float(peak.max()), 3),
                "episodes_over_threshold": int((peak > THRESHOLD).sum()),
                # The artifact detector. Near 1.0 means this concept fires because the transcript
                # changed speaker, not because of anything the model was doing.
                "fraction_at_turn_boundary": round(float(boundary.mean()), 3),
                "mean_peak_hacked": round(float(peak[hacked].mean()), 3) if hacked.any() else None,
                "mean_peak_other": round(float(peak[~hacked].mean()), 3) if (~hacked).any() else None,
            }
        entry["gap_hacked_minus_other"] = round(
            float(entry["layers"]["L25"]["mean_peak_hacked"] - entry["layers"]["L25"]["mean_peak_other"]), 3
        ) if hacked.any() and (~hacked).any() else None
        out["concepts"].append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log.info(f"wrote {args.out}")

    ranked = sorted(out["concepts"], key=lambda e: -e["layers"]["L25"]["mean_peak"])
    log.info("top 15 by L25 mean peak |z| (boundary fraction in brackets):")
    for entry in ranked[:15]:
        info = entry["layers"]["L25"]
        log.info(f"  {info['mean_peak']:5.2f} [{info['fraction_at_turn_boundary']:.2f}]  {entry['concept'][:58]}")


if __name__ == "__main__":
    main()
