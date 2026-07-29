#! /usr/bin/env python

"""Build the class catalogue the labeller is shown.

`label.py` runs in a container that holds neither `pairs.parquet` nor the screen results, so the
catalogue is rendered once here and shipped as a single small input. Each class is illustrated by
the two contrasts inside it that scored highest in the original screen, on the grounds that a
contrast already shown to be visible in generated text is the clearest way to say what the class
covers.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pyarrow.parquet as pq

log = logging.getLogger("classes")


def build(pairs: Path, hits: Path, examples: int) -> dict:
    """Render the catalogue, ordering each class's examples by screen win-rate.

    :param pairs: `pairs.parquet`, giving every pair its class, concept and antagonist.
    :param hits: `hits.json` from the original screen, giving each pair its win-rate.
    :param examples: how many contrasts to show per class.

    :return: the class names in id order, the rendered catalogue lines, and the pairs chosen.
    """
    table = pq.read_table(pairs, columns=["class_name", "concept", "antagonist"]).to_pydict()
    rate = {row["pair"]: row["rate"] for row in json.loads(hits.read_text())["rows"] if row.get("pair", -1) >= 0}

    grouped: dict[str, list[tuple[float, int, str]]] = {}
    for index, name in enumerate(table["class_name"]):
        contrast = f"{table['concept'][index]} || {table['antagonist'][index]}"
        # A pair the screen never scored sorts last rather than crashing the build.
        grouped.setdefault(name, []).append((rate.get(index, -1.0), index, contrast))

    names = sorted(grouped)
    lines, chosen = [], {}
    for identifier, name in enumerate(names):
        best = sorted(grouped[name], reverse=True)[:examples]
        lines.append(f"{identifier:3d}  {name}")
        lines += [f"       e.g. {contrast}" for _, _, contrast in best]
        chosen[name] = [{"pair": pair, "rate": round(score, 3), "contrast": contrast} for score, pair, contrast in best]

    log.info(f"{len(names)} classes, {len(lines)} catalogue lines, {len('\n'.join(lines))} characters")
    return {"classes": names, "lines": lines, "examples": chosen}


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    args.out.write_text(json.dumps(build(args.pairs, args.hits, args.examples), indent=2))
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("classes.json"))
    parser.add_argument("--pairs", type=Path, default=Path("probes-lda/pairs.parquet"))
    parser.add_argument("--hits", type=Path, default=Path("hits.json"))
    parser.add_argument("--examples", type=int, default=2)
    main(parser.parse_args())
