#! /usr/bin/env python

import json
import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

log = logging.getLogger("hits")


def resolve(record: dict[str, Any]) -> str | None:
    """Turn one judgement into whether the positive-alpha arm won, accounting for the blinding.

    :param record: one line of the judge's output.

    :return: `"positive"`, `"negative"`, `"tie"`, or None when the item is unusable.
    """
    verdict = record.get("verdict")
    if not verdict:
        return None
    broken = ("repetition_loop", "caps_lock", "word_salad", "truncated")
    if verdict.get("first_broken", "none") in broken or verdict.get("second_broken", "none") in broken:
        return None
    choice = str(verdict.get("choice", "")).strip()
    if choice == "tie":
        return "tie"
    if choice not in ("1", "2"):
        return None
    first_is_positive = not record.get("flipped", False)
    return "positive" if (choice == "1") == first_is_positive else "negative"


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    records = []
    for path in sorted(args.labels):
        records += [json.loads(line) for line in path.read_text().splitlines() if line]
    log.info(f"{len(records)} judgements from {len(args.labels)} files")

    tallies: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "tie": 0, "dropped": 0})
    meta: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record["direction"]
        outcome = resolve(record)
        tallies[name]["dropped" if outcome is None else outcome] += 1
        meta.setdefault(name, {"pair": record.get("pair", -1), "method": record.get("method", "")})

    controls = sorted(name for name in tallies if name.startswith("control_random"))
    concepts = sorted(name for name in tallies if not name.startswith("control"))
    if not controls:
        raise SystemExit("no random controls in the judgements; there is no null to test against")

    # The empirical null. With a thousand directions tested, assuming p = 0.5 would be a guess about the
    # judge's own symmetry; pooling the random directions measures it instead, and any position or
    # verbosity bias it carries lands in the null where it belongs rather than in the signal.
    won = sum(tallies[name]["positive"] for name in controls)
    lost = sum(tallies[name]["negative"] for name in controls)
    tied = sum(tallies[name]["tie"] for name in controls)
    null = won / max(1, won + lost)
    log.info(
        f"null from {len(controls)} random directions: {won}/{won + lost} = {null:.4f} decisive, "
        f"tie rate {tied / max(1, won + lost + tied):.3f}"
    )
    if abs(null - 0.5) > 0.05:
        log.warning(f"null is {null:.3f}, not near 0.5; the judge has a side preference and it is corrected for")
    floor = 1.0 / (won + lost + 2)
    if not floor <= null <= 1 - floor:
        log.warning(f"null of {null:.3f} is unresolvable from {won + lost} control judgements; clamped")
        null = min(max(null, floor), 1 - floor)

    rows = []
    for name in concepts:
        counts = tallies[name]
        decisive = counts["positive"] + counts["negative"]
        if decisive < args.minimum:
            continue
        # Two-sided: a direction whose label is simply reversed is as much a hit as one that is not, and
        # which pole the corpus called "concept" is an arbitrary naming decision.
        probability = stats.binomtest(counts["positive"], decisive, null, alternative="two-sided").pvalue
        rows.append(
            {
                "direction": name,
                "pair": meta[name]["pair"],
                "method": meta[name]["method"],
                "positive": counts["positive"],
                "negative": counts["negative"],
                "tie": counts["tie"],
                "dropped": counts["dropped"],
                "rate": counts["positive"] / decisive,
                "p": probability,
            }
        )

    order = np.argsort([row["p"] for row in rows])
    threshold = 0.0
    for rank, index in enumerate(order, start=1):
        if rows[index]["p"] <= 0.05 * rank / len(rows):
            threshold = rows[index]["p"]
    for row in rows:
        row["hit"] = bool(row["p"] <= threshold)

    hits = [row for row in rows if row["hit"]]
    log.info(
        f"{len(hits)} of {len(rows)} directions steer detectably at FDR 0.05 "
        f"({100 * len(hits) / max(1, len(rows)):.1f}%), p threshold {threshold:.2e}"
    )
    if "control_shared" in tallies:
        counts = tallies["control_shared"]
        decisive = counts["positive"] + counts["negative"]
        log.info(
            f"control_shared: {counts['positive']}/{decisive} = {counts['positive'] / max(1, decisive):.3f}, "
            f"tie rate {counts['tie'] / max(1, decisive + counts['tie']):.3f}"
        )

    dropped = sum(counts["dropped"] for counts in tallies.values())
    log.info(f"{dropped} judgements dropped as degenerate or unparseable ({100 * dropped / len(records):.1f}%)")

    args.out.write_text(
        json.dumps(
            {
                "null": null,
                "threshold": threshold,
                "tested": len(rows),
                "hits": len(hits),
                "dropped": dropped,
                "rows": sorted(rows, key=lambda row: row["p"]),
            },
            indent=2,
        )
    )
    log.info(f"wrote {args.out}")

    for row in sorted(rows, key=lambda item: item["p"])[:25]:
        log.info(
            f"  pair {row['pair']:>4} {row['positive']:>3}/{row['positive'] + row['negative']:<3} = "
            f"{row['rate']:.2f}  p={row['p']:.1e}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("labels", type=Path, nargs="+", help="labels.jsonl files from judge.py")
    parser.add_argument("--out", type=Path, default=Path("hits.json"))
    parser.add_argument("--minimum", type=int, default=8, help="decisive judgements a direction needs to be tested")
    main(parser.parse_args())
