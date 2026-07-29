#! /usr/bin/env python

"""Merge the labelling shards and report whether the result is usable.

Ten containers each labelled every tenth conversation. This stitches them back together and answers
the questions that decide whether curation can proceed: did every shard arrive, is the corpus whole,
and does each class have enough candidates to pick eight from.

A class with no coverage is a real finding, not a bug -- the prompts come from lmsys only, with no
fallback source, so subjects that public chat logs do not contain simply cannot be tested.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

log = logging.getLogger("merge")


def read(root: Path) -> tuple[list[dict], list[Path]]:
    """Collect every labelled row from every downloaded shard.

    :param root: directory holding `shard-NNN/labels.jsonl`.

    :return: the rows in shard order, and the files they came from.
    """
    files = sorted(root.rglob("labels.jsonl"))
    rows = []
    for source in files:
        count = 0
        with source.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                    count += 1
        log.info(f"  {source.parent.name}: {count} rows")
    return rows, files


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    classes = json.loads(args.classes.read_text())["classes"]

    rows, files = read(args.labels)
    log.info(f"{len(rows)} rows from {len(files)} shards")
    if len(files) != args.shards:
        log.warning(f"expected {args.shards} shards, found {len(files)} -- a shard is missing")

    seen = Counter(row["conversation_id"] for row in rows)
    duplicated = [key for key, count in seen.items() if count > 1]
    if duplicated:
        log.warning(f"{len(duplicated)} conversation ids appear more than once, e.g. {duplicated[:3]}")

    # How many classes each prompt was given. A corpus that came back almost entirely empty means the
    # labeller was too strict and every candidate pool will be thin.
    widths = Counter(len(row["labels"]) for row in rows)
    labelled = sum(count for width, count in widths.items() if width)
    log.info(f"unique conversations: {len(seen)}")
    log.info(f"labelled with at least one class: {labelled} ({100 * labelled / max(1, len(rows)):.1f}%)")
    for width in sorted(widths):
        log.info(f"  {width} classes: {widths[width]} prompts ({100 * widths[width] / max(1, len(rows)):.1f}%)")
    log.info(f"truncated at --max-chars: {sum(row.get('truncated', False) for row in rows)}")

    per_class: Counter[int] = Counter()
    strong: Counter[int] = Counter()
    for row in rows:
        for label in row["labels"]:
            per_class[label["class_id"]] += 1
            if label["score"] >= args.strong:
                strong[label["class_id"]] += 1

    empty = [index for index in range(len(classes)) if not per_class[index]]
    thin = [index for index in range(len(classes)) if 0 < strong[index] < args.needed]
    log.info(f"classes with candidates: {len(classes) - len(empty)} of {len(classes)}")
    log.info(f"classes with fewer than {args.needed} candidates at score >= {args.strong}: {len(thin) + len(empty)}")
    for index in empty:
        log.info(f"  EMPTY  {index:3d}  {classes[index]}")
    for index in sorted(thin, key=lambda i: strong[i]):
        log.info(f"  thin   {index:3d}  {strong[index]:4d} strong, {per_class[index]:5d} total  {classes[index]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    args.report.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "unique": len(seen),
                "shards": len(files),
                "duplicated": len(duplicated),
                "labelled": labelled,
                "widths": {str(width): count for width, count in sorted(widths.items())},
                "per_class": {str(index): per_class[index] for index in range(len(classes))},
                "strong_per_class": {str(index): strong[index] for index in range(len(classes))},
                "empty_classes": empty,
                "threshold": args.strong,
            },
            indent=2,
        )
    )
    log.info(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path(".bak/label"))
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--out", type=Path, default=Path(".bak/labels-all.jsonl"))
    parser.add_argument("--report", type=Path, default=Path(".bak/labels-report.json"))
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--needed", type=int, default=24, help="candidates a class needs to be pickable")
    parser.add_argument("--strong", type=float, default=0.5, help="score at which a candidate counts")
    main(parser.parse_args())
