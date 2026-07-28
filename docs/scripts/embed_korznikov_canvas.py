"""Embed korznikov stories + class catalog into korznikov-dataset.canvas.tsx."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_ROOT = ROOT.parent
REPO_ROOT = DOCS_ROOT.parent
DEFAULT_BUNDLE = DOCS_ROOT / "data" / "korznikov_stories_bundle.json"
DEFAULT_INDEX = REPO_ROOT / "01_eval/results/concept_viewer/index.json"
DEFAULT_CANVAS = DOCS_ROOT / "canvases" / "korznikov-dataset.canvas.tsx"
CURSOR_CANVAS = Path.home() / ".cursor/projects/home-User18-airi-summer-project/canvases/korznikov-dataset.canvas.tsx"
MARKER_START = "// EMBEDDED_KORZNIKOV_DATA_START"
MARKER_END = "// EMBEDDED_KORZNIKOV_DATA_END"

DATASET_MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--canvas", type=Path, default=DEFAULT_CANVAS)
    parser.add_argument(
        "--cursor-canvas",
        type=Path,
        default=CURSOR_CANVAS,
        help="Cursor IDE canvas path (synced after embed)",
    )
    return parser.parse_args()


def load_class_catalog(index_path: Path) -> list[dict]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in index["pairs"]:
        by_class[row["class_name"]].append(
            {
                "pair_id": row["pair"],
                "concept": row["concept"],
                "antagonist": row["antagonist"],
            }
        )
    classes = []
    for name in sorted(by_class):
        pairs = sorted(by_class[name], key=lambda p: p["pair_id"])
        classes.append({"name": name, "pairs": pairs})
    return classes


def embed_stories(bundle: dict) -> dict[str, dict[str, dict[str, dict[str, list[str]]]]]:
    languages = bundle.get("languages", [])
    genres = bundle.get("genres", [])
    stories: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {}
    for pair_id, by_genre in bundle["stories"].items():
        by_genre_out: dict[str, dict[str, dict[str, list[str]]]] = {}
        for genre in genres:
            by_lang: dict[str, dict[str, list[str]]] = {}
            for lang in languages:
                by_variant = by_genre.get(genre, {}).get(lang, {})
                if not by_variant:
                    continue
                variants_out = {
                    variant: tuple_
                    for variant, tuple_ in sorted(by_variant.items(), key=lambda kv: int(kv[0]))
                    if tuple_
                }
                if variants_out:
                    by_lang[lang] = variants_out
            if by_lang:
                by_genre_out[genre] = by_lang
        if by_genre_out:
            stories[pair_id] = by_genre_out
    return stories


def embed_block(bundle: dict, index_path: Path) -> str:
    if "class_name" not in bundle or "pairs" not in bundle:
        raise SystemExit(
            "Bundle must contain one class. Export with:\n  python docs/scripts/export_korznikov_stories_bundle.py"
        )

    embedded_class = bundle["class_name"]
    embedded_model = bundle["model"]
    languages = bundle.get("languages", [])
    genres = bundle.get("genres", [])
    stories = embed_stories(bundle)
    classes = load_class_catalog(index_path)

    guidance = {(p["concept"], p["antagonist"]): p.get("narrative_guidance", "") for p in bundle["pairs"]}
    for cls in classes:
        if cls["name"] == embedded_class:
            cls["pairs"] = [
                {
                    **p,
                    "narrative_guidance": guidance.get((p["concept"], p["antagonist"]), ""),
                }
                for p in cls["pairs"]
            ]
            break

    payload = {
        "models": DATASET_MODELS,
        "embedded_model": embedded_model,
        "languages": languages,
        "genres": genres,
        "variants": list(range(10)),
        "embedded_class": embedded_class,
        "classes": classes,
        "stories": stories,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    cell_count = sum(
        len(by_variant)
        for by_genre in stories.values()
        for by_lang in by_genre.values()
        for by_variant in by_lang.values()
    )

    return (
        f"{MARKER_START}\n"
        f"const KORZNIKOV_DATA = {payload_json};\n"
        f"// {len(classes)} classes · stories: {embedded_class!r} / {embedded_model!r} "
        f"({len(stories)} pairs, {cell_count} cells)\n"
        f"{MARKER_END}"
    )


def patch_canvas(canvas_path: Path, embedded: str) -> int:
    canvas = canvas_path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END), re.DOTALL)
    if not pattern.search(canvas):
        raise SystemExit(f"Markers not found in {canvas_path}")
    canvas = pattern.sub(lambda _: embedded, canvas)
    canvas_path.write_text(canvas, encoding="utf-8")
    return canvas_path.stat().st_size // 1024


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    embedded = embed_block(bundle, args.index)

    patch_canvas(args.canvas, embedded)
    size_kb = args.canvas.stat().st_size // 1024
    print(f"Updated {args.canvas} ({size_kb} KB)")

    if args.cursor_canvas != args.canvas and args.cursor_canvas.parent.exists():
        shutil.copy2(args.canvas, args.cursor_canvas)
        cursor_kb = args.cursor_canvas.stat().st_size // 1024
        print(f"Synced {args.cursor_canvas} ({cursor_kb} KB)")


if __name__ == "__main__":
    main()
