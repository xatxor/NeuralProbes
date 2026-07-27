"""Export feature_stories rows for the korznikov-dataset canvas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DOCS_ROOT = ROOT.parent
REPO_ROOT = DOCS_ROOT.parent
DEFAULT_INDEX = REPO_ROOT / "01_eval/results/concept_viewer/index.json"
DEFAULT_OUTPUT = DOCS_ROOT / "data" / "korznikov_stories_bundle.json"

DEFAULT_CLASS = "Vulnerability & Resilience"
MODEL = "deepseek/deepseek-v4-flash"
LANGS = ["English", "German", "Mandarin Chinese", "Russian", "Spanish"]
GENRES = [
    "case_study",
    "dialogue",
    "diary",
    "fable",
    "letter",
    "memo",
    "monologue",
    "narrative_3rd_person",
    "news",
    "speech",
]
CJK_LANGUAGES = {"Mandarin Chinese"}
CJK_PUNCT = "。！？，、；："


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Export all ontology classes (for class selector in canvas)",
    )
    parser.add_argument(
        "--class-name",
        default=DEFAULT_CLASS,
        help="Single class to export (ignored with --all-classes)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGS,
        help="Languages to include (default: all five dataset languages)",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        default=GENRES,
        help="Genres to include (default: all ten dataset genres)",
    )
    return parser.parse_args()


def split_story(
    concept_text: str,
    antagonist_text: str,
    language: str | None = None,
) -> tuple[str, str, str]:
    """Return shared opening and pole-specific suffixes."""
    i = 0
    limit = min(len(concept_text), len(antagonist_text))
    while i < limit and concept_text[i] == antagonist_text[i]:
        i += 1

    if language in CJK_LANGUAGES:
        if i > 0:
            for j in range(i - 1, max(-1, i - 30), -1):
                if concept_text[j] in CJK_PUNCT:
                    i = j + 1
                    break
        return concept_text[:i], concept_text[i:], antagonist_text[i:]

    while i > 0 and concept_text[i - 1] not in " \n\t":
        i -= 1
    return concept_text[:i], concept_text[i:], antagonist_text[i:]


def main() -> None:
    args = parse_args()
    t0 = time.time()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    pair_map = {(p["concept"], p["antagonist"]): p["pair"] for p in index["pairs"]}

    if args.all_classes:
        allowed_classes: set[str] | None = None
        class_pair_ids = {p["pair"] for p in index["pairs"]}
    else:
        allowed_classes = {args.class_name}
        class_pair_ids = {
            p["pair"] for p in index["pairs"] if p["class_name"] == args.class_name
        }
        if not class_pair_ids:
            raise SystemExit(f"No pairs found for class_name={args.class_name!r}")

    classes: dict[str, dict] = {}
    count = 0

    ds = load_dataset("AntonKorznikov/feature_stories", split="train", streaming=True)
    for ex in ds:
        if ex["model"] != args.model:
            continue
        class_name = ex["class_name"]
        if allowed_classes is not None and class_name not in allowed_classes:
            continue
        if ex["language"] not in args.languages:
            continue
        if ex["genre"] not in args.genres:
            continue

        pair_key = (ex["concept"], ex["antagonist"])
        pair_id = pair_map.get(pair_key)
        if pair_id is None or pair_id not in class_pair_ids:
            continue

        setup, concept_pole, antagonist_pole = split_story(
            ex["concept_text"],
            ex["antagonist_text"],
            ex["language"],
        )
        pid = str(pair_id)
        genre = ex["genre"]
        lang = ex["language"]
        variant = str(ex["pair_number"])

        class_bucket = classes.setdefault(
            class_name,
            {"pairs": [], "stories": {}, "_seen": set()},
        )
        class_bucket["stories"].setdefault(pid, {}).setdefault(genre, {}).setdefault(lang, {})[
            variant
        ] = [setup, concept_pole, antagonist_pole]

        if pid not in class_bucket["_seen"]:
            class_bucket["_seen"].add(pid)
            class_bucket["pairs"].append(
                {
                    "pair_id": pair_id,
                    "concept": ex["concept"],
                    "antagonist": ex["antagonist"],
                    "narrative_guidance": ex["narrative_guidance"],
                }
            )

        count += 1

    for class_name in classes:
        classes[class_name]["pairs"].sort(key=lambda row: row["pair_id"])
        del classes[class_name]["_seen"]

    total_pairs = sum(len(c["pairs"]) for c in classes.values())

    if args.all_classes:
        bundle = {
            "model": args.model,
            "languages": list(args.languages),
            "genres": list(args.genres),
            "classes": {name: {"pairs": c["pairs"], "stories": c["stories"]} for name, c in classes.items()},
            "stats": {
                "classes": len(classes),
                "pairs": total_pairs,
                "languages": len(args.languages),
                "genres": len(args.genres),
                "variants": 10,
                "models_in_dataset": 2,
                "rows_exported": count,
                "rows_full_dataset": 1036000,
            },
        }
    else:
        only = classes[args.class_name]
        bundle = {
            "class_name": args.class_name,
            "model": args.model,
            "languages": list(args.languages),
            "genres": list(args.genres),
            "pairs": only["pairs"],
            "stories": only["stories"],
            "stats": {
                "class_name": args.class_name,
                "pairs_in_class": len(only["pairs"]),
                "languages": len(args.languages),
                "genres": len(args.genres),
                "variants": 10,
                "models_in_dataset": 2,
                "rows_exported": count,
                "rows_full_dataset": 1036000,
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    size_mb = args.output.stat().st_size / 1e6
    if args.all_classes:
        print(
            f"Wrote {args.output} ({size_mb:.1f} MB, {count} rows, "
            f"{len(classes)} classes, {total_pairs} pairs, {elapsed:.0f}s)"
        )
    else:
        print(
            f"Wrote {args.output} ({size_mb:.1f} MB, {count} rows, "
            f"{len(only['pairs'])} pairs, class={args.class_name!r}, {elapsed:.0f}s)"
        )


if __name__ == "__main__":
    main()
