#! /usr/bin/env python

"""Answer the question the re-screen was built to ask.

The original screen tested all 1036 vectors on sixteen generic prompts and passed 790. Its failures
are ambiguous: a vector looks inert whether it encodes nothing behavioural or whether no prompt gave
it room. This compares the same vectors on prompts drawn from each one's own class, so the two
explanations come apart.

Three numbers carry the argument. The **win rate** is how often steering toward a concept produced
the response a judge called more concept-like. The **null** is what that rate comes to for random
directions on the same prompts, measured rather than assumed. And **concept_expressible** is the
judge's own verdict on whether a prompt could show the concept at all -- the field that separates
"the vector does nothing" from "the prompt could not show it".
"""

import json
import logging
import math
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

log = logging.getLogger("analyze")


def toward(row: dict) -> float | None:
    """Re-express one verdict as a lean toward the steered-positive arm.

    The judge scores `concept_lean` over the two responses as presented, negative meaning the first
    shows more of the concept. Presentation order was randomised, so the sign has to be undone
    before anything can be pooled.

    :param row: one judged comparison.

    :return: positive when the arm steered toward the concept won, or None when unusable.
    """
    verdict = row.get("verdict")
    if not verdict:
        return None
    lean = verdict.get("concept_lean")
    if lean is None:
        return None
    # first is the plus arm unless the presentation was flipped, so un-flipping restores the sign.
    return float(lean) if row.get("flipped") else -float(lean)


def interval(wins: int, total: int) -> tuple[float, float]:
    """A Wilson score interval, which stays sane on the small per-class counts here.

    A normal approximation breaks down when a class contributes a handful of comparisons, and
    several classes contribute fewer than ten.

    :param wins: comparisons the concept arm won.
    :param total: decisive comparisons.

    :return: lower and upper bounds at roughly 95%.
    """
    if not total:
        return (0.0, 1.0)
    z, rate = 1.96, wins / total
    middle = rate + z * z / (2 * total)
    spread = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return tuple(max(0.0, min(1.0, (middle + sign * spread) / (1 + z * z / total))) for sign in (-1, 1))


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    rows = [
        json.loads(line)
        for source in sorted(args.labels.rglob("labels.jsonl"))
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    log.info(f"{len(rows)} judged comparisons")

    classes = json.loads(args.classes.read_text())["classes"]
    pairs = pq.read_table(args.pairs, columns=["class_name", "concept", "antagonist"]).to_pydict()
    # The original screen's per-vector rates are only needed to say which vectors it failed that now
    # pass. Everything else -- win rates, the measured null, concept_expressible -- stands on the
    # judged data alone, so a missing file costs one comparison rather than the whole analysis.
    if args.screen.exists():
        screen = {row["pair"]: row for row in json.loads(args.screen.read_text())["rows"] if row.get("pair", -1) >= 0}
    else:
        screen = {}
        log.warning(f"{args.screen} not found; rescued/lost against the original screen is skipped")

    usable = [row for row in rows if row.get("verdict")]
    echoed = sum(1 for row in rows if row.get("echoed"))
    log.info(
        f"usable {len(usable)} ({100 * len(usable) / max(1, len(rows)):.1f}%), "
        f"echoed {echoed} ({100 * echoed / max(1, len(rows)):.1f}%)"
    )
    if echoed > len(rows) // 20:
        log.warning("the judge is reproducing its own example; these labels are not usable")

    # Whether the prompt could express the concept at all, per class. This is the diagnostic the
    # whole re-screen turns on, so it is reported before any win rate.
    room: dict[int, list[bool]] = defaultdict(list)
    for row in usable:
        room[row["class_id"]].append(bool(row["verdict"].get("concept_expressible")))
    overall = [value for values in room.values() for value in values]
    log.info(f"concept_expressible: {100 * sum(overall) / max(1, len(overall)):.1f}% overall")

    # Steered comparisons, split by layer, pooled per direction and per class.
    per_direction: dict[tuple, list[float]] = defaultdict(list)
    per_class: dict[tuple, list[float]] = defaultdict(list)
    controls: dict[tuple, list[float]] = defaultdict(list)
    shared: dict[tuple, list[float]] = defaultdict(list)
    ablation: dict[tuple, list[float]] = defaultdict(list)
    effects: dict[str, int] = defaultdict(int)

    for row in usable:
        lean = toward(row)
        if lean is None:
            continue
        key = (row["layer"], row["class_id"])
        if row["comparison"] == "ablate":
            ablation[key].append(lean)
            continue
        if row["kind"] == "control":
            # `control_shared` is the normalised mean of all 1036 vectors, not a random direction:
            # it scored 0.661 in the original screen and 0.688 here. Pooling it into the null lifts
            # that null from ~0.49 to ~0.54 and silently fails hundreds of vectors that do beat
            # chance. It is measured separately, as a second reference point.
            if row["direction"] == "control_shared":
                shared[key].append(lean)
            else:
                controls[key].append(lean)
        else:
            per_direction[(row["layer"], row["pair"])].append(lean)
            per_class[key].append(lean)
        for name in row["verdict"].get("side_effects") or []:
            effects[name] += 1

    layers = sorted({key[0] for key in per_direction})
    log.info(f"layers judged: {layers}")

    report: dict[str, dict] = {}
    for layer in layers:
        nulls = {}
        for (this, identifier), leans in controls.items():
            if this == layer:
                decisive = [value for value in leans if value != 0]
                nulls[identifier] = (
                    sum(value > 0 for value in decisive) / len(decisive) if decisive else None,
                    len(decisive),
                )
        pooled = [value for (this, _), leans in controls.items() if this == layer for value in leans if value != 0]
        global_null = sum(value > 0 for value in pooled) / max(1, len(pooled))
        pooled_shared = [v for (t, _), ls in shared.items() if t == layer for v in ls if v != 0]
        shared_rate = sum(v > 0 for v in pooled_shared) / max(1, len(pooled_shared))
        log.info(f"L{layer}: null {global_null:.4f} over {len(pooled)} random-control comparisons; "
                 f"control_shared {shared_rate:.4f} over {len(pooled_shared)}")

        beat, tested = 0, 0
        entries = {}
        for (this, pair), leans in sorted(per_direction.items()):
            if this != layer:
                continue
            decisive = [value for value in leans if value != 0]
            if not decisive:
                continue
            wins = sum(value > 0 for value in decisive)
            rate = wins / len(decisive)
            low, _ = interval(wins, len(decisive))
            identifier = classes.index(pairs["class_name"][pair])
            local = nulls.get(identifier, (None, 0))[0]
            passed = low > (local if local is not None else global_null)
            beat += passed
            tested += 1
            entries[str(pair)] = {
                "concept": pairs["concept"][pair],
                "antagonist": pairs["antagonist"][pair],
                "class": pairs["class_name"][pair],
                "rate": round(rate, 4),
                "decisive": len(decisive),
                "mean_lean": round(sum(decisive) / len(decisive), 4),
                "wilson_low": round(low, 4),
                "class_null": round(local, 4) if local is not None else None,
                "beats_null": passed,
                "screen_rate": screen.get(pair, {}).get("rate"),
                "screen_hit": screen.get(pair, {}).get("hit"),
            }

        rescued = [
            key for key, value in entries.items() if value["beats_null"] and value["screen_hit"] is False
        ]
        lost = [key for key, value in entries.items() if not value["beats_null"] and value["screen_hit"] is True]
        log.info(f"L{layer}: {beat} of {tested} vectors beat their null ({100 * beat / max(1, tested):.1f}%)")
        log.info(f"L{layer}: {len(rescued)} vectors the original screen failed now pass")
        log.info(f"L{layer}: {len(lost)} vectors the original screen passed now fail")

        report[f"L{layer}"] = {
            "global_null": round(global_null, 4),
            "control_shared": round(shared_rate, 4),
            "control_comparisons": len(pooled),
            "vectors_tested": tested,
            "vectors_beating_null": beat,
            "rescued": sorted(rescued, key=int),
            "lost": sorted(lost, key=int),
            "class_nulls": {str(k): v[0] for k, v in sorted(nulls.items()) if v[0] is not None},
            "ablation_mean": {
                str(identifier): round(sum(v) / len(v), 4)
                for (this, identifier), v in sorted(ablation.items())
                if this == layer and v
            },
            "vectors": entries,
        }

    report["expressible_by_class"] = {
        str(identifier): round(sum(values) / len(values), 4) for identifier, values in sorted(room.items()) if values
    }
    report["side_effects"] = dict(sorted(effects.items(), key=lambda item: -item[1]))
    report["judged"] = len(rows)
    report["usable"] = len(usable)
    report["echoed"] = echoed

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info(f"wrote {args.out}")
    log.info(f"side effects: {dict(list(report['side_effects'].items())[:6])}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path(".bak/verdict"), help="judged shards")
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--pairs", type=Path, default=Path("probes-lda/pairs.parquet"))
    parser.add_argument("--screen", type=Path, default=Path("hits.json"), help="the original screen's rates")
    parser.add_argument("--out", type=Path, default=Path("rescreen.json"))
    main(parser.parse_args())
