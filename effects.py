#! /usr/bin/env python

import json
import logging
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

log = logging.getLogger("effects")


def scores(path: Path, pairs: int, layers: int) -> np.ndarray:
    """Read one method's score file back into `[token, layer, pair]`.

    :param path: a `*_m<k>.parquet` written by `jailbreak.py`.
    :param pairs: number of concept pairs, to split the flattened row.
    :param layers: number of layers stored per token.

    :return: cosines as float32.
    """
    table = pq.read_table(path, columns=["scores"])
    flat = np.asarray(table["scores"].combine_chunks().flatten(), dtype=np.float32)
    return flat.reshape(table.num_rows, layers, pairs)


def pooled(path: Path, stem: str, pairs: int, layers: int, method: int, part: str) -> np.ndarray | None:
    """Average one generation's cosines over the tokens of interest.

    Prompt and response are kept separable because they answer different questions. A wrapper's effect
    on the *prompt* is the model reading the attack; its effect on the *response* is the model acting on
    it, and only the second is behaviour.

    :param path: directory of `jailbreak.py` output.
    :param stem: generation identifier.
    :param pairs: number of concept pairs.
    :param layers: number of layers stored per token.
    :param method: index of the construction to read.
    :param part: `prompt`, `response`, or `all`.

    :return: `[layer, pair]` means, or None when the generation is missing.
    """
    table = path / f"{stem}.tokens.parquet"
    block = path / f"{stem}_m{method}.parquet"
    if not table.exists() or not block.exists():
        return None
    roles = pq.read_table(table, columns=["role"])["role"].to_pylist()
    values = scores(block, pairs, layers)
    if part == "all":
        return values.mean(axis=0)
    keep = [index for index, role in enumerate(roles) if role == part]
    return values[keep].mean(axis=0) if keep else None


def grouped(dump: Path, names: Path, concepts: list[str], antagonists: list[str]) -> dict[str, list[int]]:
    """Resolve the hand-named concept groups back to concrete pair ids.

    The groups were read off the clustering by hand and recorded as names against cluster ids per
    construction; the dump holds each cluster's members as "concept / antagonist" strings. Joining the
    two turns a prediction written in English into a set of rows that can be tested. A name appearing
    under several constructions contributes the union, since agreement across constructions was the
    reason those groups were trusted in the first place.

    :param dump: `{panel: {cluster: ["concept / antagonist", ...]}}` from `$umap.py --dump`.
    :param names: `{panel: {cluster: name}}` written by hand.
    :param concepts: concept pole of every pair, in row order.
    :param antagonists: antagonist pole of every pair, in row order.

    :return: group name to the pair ids belonging to it.
    """
    lookup = {
        f"{concept} / {antagonist}": index for index, (concept, antagonist) in enumerate(zip(concepts, antagonists))
    }
    clusters = json.loads(dump.read_text())
    labelled = json.loads(names.read_text())
    groups: dict[str, set[int]] = defaultdict(set)
    for panel, mapping in labelled.items():
        if panel.startswith("_") or not isinstance(mapping, dict):
            continue
        for cluster, name in mapping.items():
            for member in clusters.get(panel, {}).get(cluster, []):
                if (row := lookup.get(member)) is not None:
                    groups[name].add(row)
    return {name: sorted(rows) for name, rows in groups.items() if rows}


def tokenwise(path: Path, rows: list[dict[str, Any]], pairs: int, layers: int, method: int, part: str) -> np.ndarray:
    """Standard deviation of each concept's cosine over the tokens of the benign-plain cell.

    This is the denominator the viewer uses, so an effect quoted here and a token inspected on screen
    are in the same units. Accumulated as sums and sums of squares rather than by concatenating every
    token of 160 generations, which would be gigabytes before it was reduced.

    :param path: directory of `jailbreak.py` output.
    :param rows: the cell C index rows.
    :param pairs: number of concept pairs.
    :param layers: number of layers stored per token.
    :param method: index of the construction to read.
    :param part: `prompt`, `response`, or `all`.

    :return: `[layer, pair]` standard deviations.
    """
    total = np.zeros((layers, pairs), dtype=np.float64)
    square = np.zeros((layers, pairs), dtype=np.float64)
    counted = 0
    for row in rows:
        table = path / f"{row['stem']}.tokens.parquet"
        block = path / f"{row['stem']}_m{method}.parquet"
        if not table.exists() or not block.exists():
            continue
        roles = pq.read_table(table, columns=["role"])["role"].to_pylist()
        values = scores(block, pairs, layers).astype(np.float64)
        if part != "all":
            keep = [index for index, role in enumerate(roles) if role == part]
            if not keep:
                continue
            values = values[keep]
        total += values.sum(axis=0)
        square += (values**2).sum(axis=0)
        counted += len(values)
    mean = total / max(1, counted)
    return np.sqrt(np.maximum(square / max(1, counted) - mean**2, 0.0))


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    layers = [11, 14, 18, 22, 25]

    rows = [json.loads(line) for line in (args.input / "index.jsonl").read_text().splitlines() if line]
    labels = {}
    if args.labels and args.labels.exists():
        for line in args.labels.read_text().splitlines():
            if line:
                record = json.loads(line)
                labels[record.get("stem", "")] = record.get("verdict") or {}
    log.info(f"{len(rows)} generations, {len(labels)} with an outcome label")

    names = pq.read_table(args.pairs, columns=["concept", "antagonist"]).to_pydict()
    pairs = len(names["concept"])

    # Every generation pooled once, then averaged within a (behaviour, cell) so each behaviour weighs the
    # same regardless of how many of its samples survived.
    cells: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        value = pooled(args.input, row["stem"], pairs, len(layers), args.method, args.part)
        if value is not None:
            cells[(row["behaviour"], row["cell"])].append(value)
    log.info(f"pooled {sum(len(v) for v in cells.values())} generations, {time.monotonic() - started:.0f}s")

    behaviours = sorted({behaviour for behaviour, _ in cells})
    complete = [b for b in behaviours if all((b, cell) in cells for cell in "ABCD")]
    log.info(f"{len(complete)} of {len(behaviours)} behaviours have all four cells")
    if not complete:
        raise SystemExit("no behaviour has all four cells; there is nothing to difference")

    position = layers.index(args.layer)
    means = {
        (behaviour, cell): np.mean(cells[(behaviour, cell)], axis=0)[position]
        for behaviour in complete
        for cell in "ABCD"
    }

    # Reported in the same sigma the viewer shows: the spread of cosines across the *tokens* of the
    # benign-plain cell, which is the denominator `blobs.py` z-scores by. Dividing instead by the spread
    # of per-behaviour means -- an earlier version of this -- uses a denominator smaller by roughly the
    # square root of the tokens in a behaviour, which inflates every effect and makes the numbers
    # incomparable to anything on screen. The t-statistic is unchanged either way, since scaling every
    # behaviour by one constant cannot move it; what changes is whether the magnitude can be read.
    spread = tokenwise(
        args.input, [row for row in rows if row["cell"] == "C"], pairs, len(layers), args.method, args.part
    )[position]
    spread = np.maximum(spread, 1e-6)

    # The estimand. B - A is the wrapper's effect on a harmful request, but a persona wrapper raises
    # roleplay concepts whatever the payload; D - C measures exactly that nuisance with the attack
    # removed. The double difference is what appears only when the payload is harmful.
    harmful = np.array([(means[(b, "B")] - means[(b, "A")]) / spread for b in complete])
    benign = np.array([(means[(b, "D")] - means[(b, "C")]) / spread for b in complete])
    difference = harmful - benign

    # Paired across behaviours: each behaviour contributes one number per concept, so the test is over
    # 40 independent behaviours rather than over tokens, which are not independent at all.
    statistic, probability = stats.ttest_1samp(difference, 0.0, axis=0)
    order = np.argsort(probability)
    threshold = 0.0
    for rank, index in enumerate(order, start=1):
        if probability[index] <= 0.05 * rank / pairs:
            threshold = probability[index]

    effect = difference.mean(axis=0)
    table = [
        {
            "pair": index,
            "concept": names["concept"][index],
            "antagonist": names["antagonist"][index],
            "did": float(effect[index]),
            "harmful": float(harmful.mean(axis=0)[index]),
            "benign": float(benign.mean(axis=0)[index]),
            "t": float(statistic[index]),
            "p": float(probability[index]),
            "hit": bool(probability[index] <= threshold),
        }
        for index in range(pairs)
    ]
    hits = [row for row in table if row["hit"]]
    log.info(
        f"L{args.layer} {args.part} tokens: {len(hits)} of {pairs} concepts move at FDR 0.05, "
        f"p threshold {threshold:.2e}"
    )

    for row in sorted(table, key=lambda item: -abs(item["did"]))[:20]:
        mark = "*" if row["hit"] else " "
        log.info(
            f" {mark} {row['concept'][:38]:<38} did {row['did']:+.2f}s  "
            f"(B-A {row['harmful']:+.2f}s, D-C {row['benign']:+.2f}s)  p={row['p']:.1e}"
        )

    # A concept that moves under one construction at one layer may be moving because that construction
    # is dominated by the shared component, which is exactly what the six constructions exist to tell
    # apart. Agreement across them is the evidence; a single slice is an anecdote.
    if args.sweep:
        agree: dict[int, int] = defaultdict(int)
        signs: dict[int, set[int]] = defaultdict(set)
        for method in range(6):
            for other in layers:
                spot = layers.index(other)
                try:
                    pooled_cells = {
                        (behaviour, cell): np.mean(
                            [
                                value
                                for row in rows
                                if row["behaviour"] == behaviour
                                and row["cell"] == cell
                                and (value := pooled(args.input, row["stem"], pairs, len(layers), method, args.part))
                                is not None
                            ],
                            axis=0,
                        )[spot]
                        for behaviour in complete
                        for cell in "ABCD"
                    }
                except (ValueError, IndexError):
                    continue
                slice_harmful = np.array([pooled_cells[(b, "B")] - pooled_cells[(b, "A")] for b in complete])
                slice_benign = np.array([pooled_cells[(b, "D")] - pooled_cells[(b, "C")] for b in complete])
                _, slice_p = stats.ttest_1samp(slice_harmful - slice_benign, 0.0, axis=0)
                slice_effect = (slice_harmful - slice_benign).mean(axis=0)
                for index in np.flatnonzero(slice_p <= 0.05 / pairs):
                    agree[int(index)] += 1
                    signs[int(index)].add(int(np.sign(slice_effect[index])))
                log.info(f"  m{method} L{other}: {int((slice_p <= 0.05 / pairs).sum())} concepts at Bonferroni")
        stable = [index for index, count in agree.items() if count >= 24 and len(signs[index]) == 1]
        log.info(
            f"agreement: {len(stable)} concepts move consistently in >=24 of 30 slices with one sign; "
            f"{len(agree)} move in at least one"
        )
        for index in sorted(stable, key=lambda row: -agree[row])[:15]:
            log.info(f"  {names['concept'][index][:44]:<44} {agree[index]}/30 slices")

    if args.dump and args.names and args.dump.exists() and args.names.exists():
        groups = grouped(args.dump, args.names, names["concept"], names["antagonist"])
        log.info(f"{len(groups)} named groups resolved to pair ids")
        for name, members in sorted(groups.items(), key=lambda item: -abs(effect[item[1]].mean())):
            share = sum(1 for row in members if table[row]["hit"]) / len(members)
            log.info(
                f"  {name[:44]:<44} {len(members):>3} pairs  did {effect[members].mean():+.2f}s  "
                f"{100 * share:.0f}% clear FDR"
            )

    outcomes: dict[str, int] = defaultdict(int)
    for row in rows:
        verdict = labels.get(row["stem"], {})
        if row["cell"] == "B" and verdict:
            outcomes[str(verdict.get("refusal", "unlabelled"))] += 1
    if outcomes:
        log.info(f"cell B outcomes: {dict(outcomes)}")

    args.out.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "method": args.method,
                "part": args.part,
                "behaviours": len(complete),
                "threshold": float(threshold),
                "hits": len(hits),
                "outcomes": dict(outcomes),
                "rows": sorted(table, key=lambda item: item["p"]),
            },
            indent=2,
        )
    )
    # A compact copy beside the viewer's data, so the rail can rank concepts by the effect and the
    # claim can be checked against the text rather than only read off a table.
    if args.scope:
        args.scope.write_text(
            json.dumps(
                {
                    "layer": args.layer,
                    "part": args.part,
                    "threshold": float(threshold),
                    "did": {str(row["pair"]): round(row["did"], 3) for row in table},
                    "hit": [row["pair"] for row in table if row["hit"]],
                }
            )
        )
        log.info(f"wrote {args.scope} for the viewer")
    log.info(f"wrote {args.out} in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("jail"), help="jailbreak.py output directory")
    parser.add_argument("--labels", type=Path, help="outcome judgements to summarise alongside")
    parser.add_argument("--pairs", type=Path, default=Path("probes-lda/pairs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("effects.json"))
    parser.add_argument("--layer", type=int, default=18, choices=[11, 14, 18, 22, 25])
    parser.add_argument("--sweep", action="store_true", help="every layer and construction, and their agreement")
    parser.add_argument("--method", type=int, default=0, help="construction index, 0 is diff")
    parser.add_argument("--part", default="response", choices=["prompt", "response", "all"])
    parser.add_argument("--scope", type=Path, help="also write a compact copy for the viewer")
    parser.add_argument("--dump", type=Path, help="cluster membership from $umap.py --dump")
    parser.add_argument("--names", type=Path, help="hand-written names for those clusters")
    main(parser.parse_args())
