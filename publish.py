#! /usr/bin/env python

"""Publish the class labels to the Hub as a dataset.

Only the labels are published, keyed by `conversation_id`. The prompt text stays where it came from:
this is an annotation layer over lmsys-chat-1m, and re-hosting a million real user conversations
would inherit that dataset's licence for no benefit. A consumer joins on the id in one line.

The shape is wide -- one row per conversation, three fixed label slots. Rank is the slot number, so
only the id and the score carry information.

The dataset card is `hf-card.md`, copied byte for byte rather than rendered from a template.
"""

import json
import logging
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("publish")



def widen(rows: list[dict], top: int) -> pa.Table:
    """Turn per-prompt label lists into fixed columns, one slot per rank.

    :param rows: merged label rows, each with a `labels` list ordered by score.
    :param top: how many slots to emit.

    :return: the wide table, prompts with no labels already dropped.
    """
    kept = [row for row in rows if row["labels"]]
    columns: dict[str, list] = {"conversation_id": [row["conversation_id"] for row in kept]}
    for slot in range(top):
        columns[f"class_{slot + 1}_id"] = [
            row["labels"][slot]["class_id"] if slot < len(row["labels"]) else None for row in kept
        ]
        columns[f"class_{slot + 1}_score"] = [
            row["labels"][slot]["score"] if slot < len(row["labels"]) else None for row in kept
        ]

    schema = pa.schema(
        [("conversation_id", pa.string())]
        + [
            field
            for slot in range(top)
            for field in ((f"class_{slot + 1}_id", pa.int16()), (f"class_{slot + 1}_score", pa.float32()))
        ]
    )
    log.info(f"{len(kept)} of {len(rows)} prompts carry at least one label")
    return pa.table(columns, schema=schema)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    built = json.loads(args.classes.read_text())
    rows = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    table = widen(rows, args.top)
    pq.write_table(table, args.out / "labels.parquet", compression="zstd")

    examples = built.get("examples", {})
    pq.write_table(
        pa.table(
            {
                "class_id": pa.array(range(len(built["classes"])), pa.int16()),
                "class_name": pa.array(built["classes"], pa.string()),
                "example_1": pa.array(
                    [(examples.get(name) or [{}])[0].get("contrast", "") for name in built["classes"]], pa.string()
                ),
                "example_2": pa.array(
                    [
                        (examples.get(name) or [{}, {}])[1].get("contrast", "")
                        if len(examples.get(name, [])) > 1
                        else ""
                        for name in built["classes"]
                    ],
                    pa.string(),
                ),
            }
        ),
        args.out / "classes.parquet",
        compression="zstd",
    )

    # The card is hand-written and copied byte for byte. It was templated once, and regenerating it
    # kept undoing the author's line breaks, so the file on disk is now the source of truth. Its
    # figures are therefore fixed: if the labelling is ever rerun, the numbers in it must be checked
    # by hand rather than trusted to update themselves.
    if not args.card.exists():
        raise SystemExit(f"{args.card} is missing; publishing without a dataset card is not allowed")
    shutil.copyfile(args.card, args.out / "README.md")

    stale = f"{table.num_rows:,}" not in args.card.read_text()
    if stale:
        log.warning(
            f"the card does not mention {table.num_rows:,} rows; it was written for a different "
            f"labelling run and its figures need rechecking before this is published"
        )
    log.info(f"staged {args.out}: {table.num_rows} rows, {len(built['classes'])} classes")

    if args.dry_run:
        log.info("dry run, nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(args.out), repo_id=args.repo, repo_type="dataset")
    log.info(f"published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path(".bak/labels-all.jsonl"))
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--out", type=Path, default=Path(".bak/hf-dataset"))
    parser.add_argument("--repo", default="josephofthebread/lmsys-chat-1m-concept-classes")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--card", type=Path, default=Path("hf-card.md"), help="copied verbatim")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="stage the files without uploading")
    main(parser.parse_args())
