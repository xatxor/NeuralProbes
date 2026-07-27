#! /usr/bin/env python

import json
import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

log = logging.getLogger("blobs")


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


def baseline(paths: list[Path], pairs: int, layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of every concept over the benign-plain tokens.

    :param paths: the cell C score files for one method.
    :param pairs: number of concept pairs.
    :param layers: number of layers stored per token.

    :return: mean and standard deviation, each `[layer, pair]` float32.
    """
    total = np.zeros((layers, pairs), dtype=np.float64)
    square = np.zeros((layers, pairs), dtype=np.float64)
    counted = 0
    for path in paths:
        block = scores(path, pairs, layers).astype(np.float64)
        total += block.sum(axis=0)
        square += (block**2).sum(axis=0)
        counted += len(block)
    mean = total / max(1, counted)
    # The floor stops a concept that never varies in cell C from turning rounding noise into a huge z.
    deviation = np.sqrt(np.maximum(square / max(1, counted) - mean**2, 0.0))
    return mean.astype(np.float32), np.maximum(deviation, 1e-4).astype(np.float32)


def neighbours(unit: np.ndarray, count: int) -> list[list[int]]:
    """For each concept, the most similar other concepts.

    :param unit: unit directions at one layer as `[pair, hidden]`.
    :param count: neighbours to keep per concept.

    :return: neighbour indices, nearest first, excluding the concept itself.
    """
    gram = unit @ unit.T
    np.fill_diagonal(gram, -np.inf)
    order = np.argsort(-np.abs(gram), axis=1)[:, :count]
    return [row.tolist() for row in order]


def quantise(block: np.ndarray, span: float) -> tuple[bytes, float]:
    """Pack z-scores into one byte each.

    :param block: z-scores as `[token, pair]`.
    :param span: the sigma value that saturates the byte range.

    :return: the packed bytes and the scale needed to recover z.
    """
    scale = span / 127.0
    return np.clip(np.rint(block / scale), -127, 127).astype(np.int8).tobytes(), scale


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    layers = [11, 14, 18, 22, 25]
    methods = (
        "diff",
        "concept_centered",
        "antagonist_centered",
        "whitened_diff",
        "whitened_concept_centered",
        "whitened_antagonist_centered",
    )

    indexes = sorted(args.input.glob("index-*.jsonl")) or [args.input / "index.jsonl"]
    rows = [json.loads(line) for path in indexes for line in path.read_text().splitlines() if line]
    log.info(f"{len(indexes)} index file(s)")
    log.info(f"{len(rows)} generations from {args.input}")

    stacked = load_file(args.readouts)["readouts"]
    pairs = stacked.shape[2]
    labels = pq.read_table(args.pairs, columns=["pair", "concept", "antagonist", "class_name"]).to_pydict()

    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)

    unit = stacked[0, layers.index(18)]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    (data / "concepts.json").write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair": labels["pair"][row],
                        "concept": labels["concept"][row],
                        "antagonist": labels["antagonist"][row],
                        "class": labels["class_name"][row],
                    }
                    for row in range(pairs)
                ],
                "neighbours": neighbours(unit, 12),
            }
        )
    )

    written = 0
    for index, method in enumerate(methods):
        present = [row for row in rows if (args.input / f"{row['stem']}_m{index}.parquet").exists()]
        controls = [
            args.input / f"{row['stem']}_m{index}.parquet"
            for row in present
            if row["cell"] == "C" and not row.get("steer")
        ]
        if not controls:
            raise SystemExit(f"no cell C generations for {method}; there is nothing to z-score against")
        mean, deviation = baseline(controls, pairs, len(layers))
        log.info(f"{method}: baseline over {len(controls)} benign-plain generations")

        for row in present:
            block = (scores(args.input / f"{row['stem']}_m{index}.parquet", pairs, len(layers)) - mean) / deviation
            for position, layer in enumerate(layers):
                packed, scale = quantise(block[:, position], 8.0)
                (data / f"{row['stem']}.m{index}.L{layer}.bin").write_bytes(packed)
                row.setdefault("scales", {})[f"m{index}.L{layer}"] = scale
            written += 1
        log.info(f"{method}: {len(present)} generations packed, {time.monotonic() - started:.0f}s")

    for row in rows:
        table = pq.read_table(args.input / f"{row['stem']}.tokens.parquet")
        (data / f"{row['stem']}.tokens.json").write_text(
            json.dumps(
                {
                    "pieces": table["piece"].to_pylist(),
                    "role": table["role"].to_pylist(),
                    "logprob": [None if value != value else round(value, 3) for value in table["logprob"].to_pylist()],
                    "norms": {
                        f"L{layer}": [round(value, 1) for value in table[f"norm_L{layer}"].to_pylist()]
                        for layer in layers
                    },
                }
            )
        )

    verdicts = {}
    if args.labels and args.labels.exists():
        for line in args.labels.read_text().splitlines():
            if line:
                record = json.loads(line)
                verdicts[record.get("stem", "")] = record.get("verdict")

    (data / "manifest.json").write_text(
        json.dumps(
            {
                "methods": list(methods),
                "layers": layers,
                "pairs": pairs,
                "span": 8.0,
                "generations": [
                    {
                        "stem": row["stem"],
                        "behaviour": row["behaviour"],
                        "half": row["half"],
                        "topic": row["topic"],
                        "tactic": row["tactic"],
                        "cell": row["cell"],
                        "sample": row["sample"],
                        "steer": row.get("steer", ""),
                        "tokens": row["tokens"],
                        "width": row["width"],
                        "prompt": row["prompt"],
                        "scales": row.get("scales", {}),
                        "verdict": verdicts.get(row["stem"]),
                    }
                    for row in rows
                ],
            }
        )
    )
    log.info(f"wrote {data}: {written} generation-methods, {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("jail"), help="jailbreak.py output directory")
    parser.add_argument("--out", type=Path, default=Path("scope"), help="viewer directory; blobs go in its data/")
    parser.add_argument("--readouts", type=Path, default=Path("readouts.safetensors"))
    parser.add_argument("--pairs", type=Path, default=Path("probes-lda/pairs.parquet"))
    parser.add_argument("--labels", type=Path, help="judge output to attach to each generation")
    main(parser.parse_args())
