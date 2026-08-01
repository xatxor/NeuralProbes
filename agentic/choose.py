#! /usr/bin/env python

"""Choose which concept pairs to re-extract in gradient space.

Selecting the first 128 rows would sample whatever order the ontology happens to be stored in, and
the ontology is grouped by class, so a prefix would cover a handful of classes exhaustively and the
rest not at all. Two constraints instead:

**Coverage.** Sample across the 148 ontology classes proportionally, so the resulting set is a fair
miniature of the whole rather than a corner of it. Comparing gradient-space against activation-space
vectors only means something if the pairs compared are representative.

**Relevance.** Force in the concepts this project already has evidence about, whatever the sampler
says: the impasse/desperation direction that survived the audit, the evaluation-awareness pair whose
per-turn trajectory was measured, and the scope-restraint block that topped the predictive window.
Those are the ones whose gradient-space counterparts can be checked against something known.
"""

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger("choose")

VECTORS = "josephofthebread/Qwen3-8B-concept-vectors"
# Pairs this project has already measured something about, so a re-extraction is falsifiable.
# 376/316: self-reported impasse, the one concept that survived the artifact audit (316 is 376
# sign-flipped). 463: honest metacognition about being evaluated, whose per-turn trace was measured.
# 258: respectful pushback, the other audit survivor. The rest are the scope-restraint block that
# topped the shape-controlled predictive window.
ANCHORS = (376, 316, 463, 258, 611, 129, 548, 272, 519, 1014, 251, 200, 181, 906, 584, 209, 270, 296)


def find(name: str) -> str:
    """Locate a file inside the local HF cache.

    :param name: the file's name within the vectors repo.

    :return: an absolute path.
    """
    pattern = f"**/models--{VECTORS.replace('/', '--')}/snapshots/*/{name}"
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / pattern), recursive=True)
    if not matches:
        raise SystemExit(f"{name} is not in the local cache")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("pairs-128.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row["class_name"]].append(index)

    picked = {p for p in ANCHORS if p < len(rows)}
    log.info(f"{len(rows)} pairs across {len(by_class)} classes; {len(picked)} anchors forced in")

    # Proportional allocation, at least one per class while budget lasts, so no class is invisible.
    classes = sorted(by_class, key=lambda c: -len(by_class[c]))
    remaining = args.count - len(picked)
    for name in classes:
        if remaining <= 0:
            break
        pool = [i for i in by_class[name] if i not in picked]
        if not pool:
            continue
        share = max(1, round(args.count * len(by_class[name]) / len(rows)))
        take = min(share, len(pool), remaining)
        for index in rng.choice(pool, size=take, replace=False):
            picked.add(int(index))
        remaining = args.count - len(picked)

    # Top up at random if proportional rounding left the set short.
    pool = [i for i in range(len(rows)) if i not in picked]
    while len(picked) < args.count and pool:
        picked.add(int(rng.choice(pool)))
        pool = [i for i in range(len(rows)) if i not in picked]

    final = sorted(picked)[: args.count]
    covered = {rows[i]["class_name"] for i in final}
    args.out.write_text(json.dumps(final))
    log.info(f"wrote {args.out}: {len(final)} pairs covering {len(covered)}/{len(by_class)} classes")
    for index in final[:10]:
        log.info(f"  {index:>4} [{rows[index]['class_name']}] "
                 f"{rows[index]['concept']} || {rows[index]['antagonist']}")


if __name__ == "__main__":
    main()
