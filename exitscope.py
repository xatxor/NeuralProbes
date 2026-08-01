#! /usr/bin/env python

"""Pack the six ended conversations into the data the `scope/` viewer reads.

Same contract as `evalscope.py` -- a manifest, a concept catalogue, one tokens file per conversation
and one int8 blob per conversation per layer. These six are the only ones in 450 generations where
the model chose to end the conversation, so the set is small enough to read one by one.

Values are per-conversation z-scores: a concept is shown relative to how it behaved across the rest
of that same exchange, which is what stops the large constant offset every direction carries from
dominating the shading.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

from evalscope import LAYERS, SPAN, neighbours, quantise

log = logging.getLogger("exitscope")

# endtool/exits use three roles; "response" is the one app.js styles specially.
NAMES = {0: "template", 1: "user", 2: "response"}


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(args.pairs).to_pylist()
    block = load_file(args.vectors)["diff"]
    unit = block[4]                                     # L25 row of [11, 14, 18, 22, 25]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    (args.out / "concepts.json").write_text(json.dumps({
        "pairs": [{"pair": i, "concept": r["concept"], "antagonist": r["antagonist"],
                   "class": r["class_name"]} for i, r in enumerate(rows)],
        "neighbours": neighbours(unit.astype(np.float32), args.neighbours)}))

    meta = [json.loads(line) for line in (args.dir / "exits.jsonl").open()]
    data = load_file(str(args.dir / "features.safetensors"))
    scores, roles, ids, edges = data["scores"], data["roles"], data["ids"], data["edges"]

    generations = []
    for index, row in enumerate(meta):
        if not row["reproduced"]:
            log.warning(f"skipping {row['label']}: the replay did not reproduce the ending")
            continue
        lo, hi = int(edges[index]), int(edges[index + 1])
        plane = scores[lo:hi].astype(np.float32)
        z = (plane - plane.mean(axis=0, keepdims=True)) / np.maximum(plane.std(axis=0, keepdims=True), 1e-6)

        which = "greedy" if row["pass"] == "greedy" else f"seed{row['seed']}"
        name = f"exit{index + 1}-lvl{row['level']}-{row['protocol']}-{which}"
        scales = {}
        for slot, layer in enumerate(LAYERS):
            packed, scale = quantise(z[:, slot, :], SPAN)
            (args.out / f"{name}.m0.L{layer}.bin").write_bytes(packed)
            scales[f"m0.L{layer}"] = scale
        (args.out / f"{name}.tokens.json").write_text(json.dumps({
            "pieces": tokenizer.convert_ids_to_tokens([int(v) for v in ids[lo:hi]]),
            "role": [NAMES[int(r)] for r in roles[lo:hi]],
            "logprob": [None] * (hi - lo)}))

        generations.append({
            "stem": name, "ending": f"level {row['level']}",
            "outcome": f"ENDED at turn {row['exit_turn']} · {row['new_reason'][:70]}",
            "seed": row["seed"], "variant": row["variant"], "turns": len(row["turns"]),
            "distinct": row["exit_turn"], "tokens": hi - lo, "width": len(rows), "scales": scales})
        log.info(f"{name}: {hi - lo} tokens, ended turn {row['exit_turn']}")

    # The all-caps level 5 exit is the only one motivated by conduct rather than by the model
    # inferring that the user asked to stop, so it leads.
    generations.sort(key=lambda r: (r["ending"] != "level 5", r["stem"]))
    (args.out / "manifest.json").write_text(json.dumps({
        "methods": ["z-score (per conversation)"], "layers": list(LAYERS),
        "pairs": len(rows), "span": SPAN, "generations": generations}))
    log.info(f"wrote {len(generations)} conversations to {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path(".bak/exits"))
    parser.add_argument("--out", type=Path, default=Path(".bak/exitscope"))
    parser.add_argument("--pairs", type=Path, default=Path(".bak/probes/pairs.parquet"))
    parser.add_argument("--vectors", type=Path, default=Path(".bak/probes/diff.safetensors"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--neighbours", type=int, default=12)
    main(parser.parse_args())
