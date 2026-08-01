#! /usr/bin/env python

"""Pack the three evaluations into the data the `scope/` viewer already reads.

The viewer's contract is unchanged from the agentic and jailbreak versions -- a manifest, a concept
catalogue, one tokens file per generation and one int8 blob per generation per layer -- so this only
has to speak that format. Only the descriptive fields differ: a generation here is described by its
task, its decoding pass, and whether it got the right answer.

Values are per-generation z-scores, which is what makes one response readable on its own: a concept
is shown relative to how it behaved across the rest of that same response, so the large constant
offset every direction carries cancels instead of dominating.

The token ids were not stored by `evals.py`, so they are reconstructed here and CHECKED rather than
assumed: the prompt is rebuilt from the stimulus files, the reply is re-tokenised, and the total must
equal the token count the features were computed over. A generation whose reconstruction does not
match exactly is dropped rather than shown against tokens it might not correspond to.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

from evals import GENERATED, GIVEN, TEMPLATE, USER, build, items, pieces

log = logging.getLogger("evalscope")

LAYERS = (18, 25)
# Clipping bound for the int8 packing, in z units. Beyond 8 sigma is vanishingly rare and clipping
# costs nothing, while a wider span would waste resolution on the range that matters.
SPAN = 8.0
# "response" is the one role app.js styles specially; the rest are shown as plain labels.
NAMES = {TEMPLATE: "template", USER: "user", GIVEN: "given", GENERATED: "response"}


def quantise(block: np.ndarray, span: float) -> tuple[bytes, float]:
    """Pack a `[token, pair]` plane of z-scores into int8 with a shared scale.

    :param block: z-scores for one layer.
    :param span: clip bound in units of the score.

    :return: the packed bytes and the scale that turns them back into z.
    """
    scale = span / 127.0
    return np.clip(np.round(block / scale), -127, 127).astype(np.int8).tobytes(), scale


def neighbours(unit: np.ndarray, count: int) -> list[list[int]]:
    """Nearest concepts by cosine, for the viewer's "similar concepts" affordance.

    :param unit: unit-normalised directions as `[pair, hidden]`.
    :param count: how many neighbours per concept.

    :return: neighbour pair ids per concept.
    """
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -2.0)
    return np.argsort(-similarity, axis=1)[:, :count].tolist()


def stem(row: dict) -> str:
    """Name one generation `<task>-<pass>-<kind>-<label>`.

    :param row: a record from `replies.jsonl`.

    :return: a filename stem, safe for a URL and sorting sensibly.
    """
    which = "greedy" if row["pass"] == "greedy" else f"seed{row['seed']}"
    label = "".join(ch for ch in row["label"] if ch.isalnum())
    return f"{row['task']}-{which}-{row['kind']}-{label}"


def restore(row: dict, turns: list, part: dict, tokenizer, total: int) -> list[int] | None:
    """Rebuild one generation's token ids and verify they match the stored features.

    `evals.py` saved features and roles but not ids. The prompt is deterministic given the stimulus,
    and the reply is recoverable by re-tokenising its text -- except that the reply was decoded with
    `skip_special_tokens=True`, so a trailing `<|im_end|>` is missing whenever the model stopped on
    its own rather than hitting the token budget.

    :param row: the reply record.
    :param turns: the conversation that produced it.
    :param part: template scaffolding from `pieces`.
    :param tokenizer: the model's tokenizer.
    :param total: how many tokens the stored features cover.

    :return: the ids, or None if the reconstruction does not land exactly on `total`.
    """
    ids, _ = build(turns, part, tokenizer)
    reply = tokenizer(row["reply"], add_special_tokens=False)["input_ids"]
    if len(ids) + len(reply) + 1 == total:
        reply = reply + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    if len(ids) + len(reply) != total:
        return None
    return ids + reply


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    part = pieces(tokenizer)
    every = items(args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(args.pairs).to_pylist()
    block = load_file(args.vectors)["diff"]
    # Row 4 of the file's layer axis [11, 14, 18, 22, 25] is L25, the deeper of the two read here.
    unit = block[4]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    (args.out / "concepts.json").write_text(json.dumps({
        "pairs": [{"pair": i, "concept": r["concept"], "antagonist": r["antagonist"],
                   "class": r["class_name"]} for i, r in enumerate(rows)],
        "neighbours": neighbours(unit.astype(np.float32), args.neighbours),
    }))
    log.info(f"concepts.json: {len(rows)} pairs, {args.neighbours} neighbours each")

    generations, dropped = [], 0
    for shard in sorted(args.dir.glob("shard-*")):
        meta = [json.loads(line) for line in (shard / "replies.jsonl").open()]
        data = load_file(str(shard / "features.safetensors"))
        scores, roles, edges = data["scores"], data["roles"], data["edges"]
        for index, row in enumerate(meta):
            lo, hi = int(edges[index]), int(edges[index + 1])
            ids = restore(row, every[row["item"]]["turns"], part, tokenizer, hi - lo)
            if ids is None:
                dropped += 1
                log.warning(f"dropped {stem(row)}: reconstruction did not match {hi - lo} tokens")
                continue

            plane = scores[lo:hi].astype(np.float32)                     # [token, layer, pair]
            # Z-scored within this generation, per concept, exactly as the viewer's other packers do.
            centre = plane.mean(axis=0, keepdims=True)
            spread = plane.std(axis=0, keepdims=True)
            z = (plane - centre) / np.maximum(spread, 1e-6)

            name = stem(row)
            scales = {}
            for slot, layer in enumerate(LAYERS):
                packed, scale = quantise(z[:, slot, :], SPAN)
                (args.out / f"{name}.m0.L{layer}.bin").write_bytes(packed)
                scales[f"m0.L{layer}"] = scale
            (args.out / f"{name}.tokens.json").write_text(json.dumps({
                "pieces": tokenizer.convert_ids_to_tokens(ids),
                "role": [NAMES[int(r)] for r in roles[lo:hi]],
                "logprob": [None] * len(ids),
            }))

            verdict = {None: "—", True: "correct", False: "WRONG"}[row["correct"]]
            generations.append({
                "stem": name, "ending": row["task"],
                "outcome": f"{verdict} · {row['reply'][:60]}".strip(),
                "seed": row["seed"], "variant": row["kind"],
                "turns": sum(1 for r, _ in every[row["item"]]["turns"] if r == "user"),
                "distinct": row["answer"], "tokens": len(ids), "width": len(rows),
                "scales": scales,
            })
        log.info(f"{shard.name}: {len(meta)} generations")

    # Wrong answers first within each task, so the failures are what you land on rather than having
    # to hunt for them.
    order = {"letters": 0, "states": 1, "hostile": 2}
    generations.sort(key=lambda r: (order.get(r["ending"], 9), r["outcome"].startswith("correct"), r["stem"]))
    (args.out / "manifest.json").write_text(json.dumps({
        "methods": ["z-score (per generation)"], "layers": list(LAYERS),
        "pairs": len(rows), "span": SPAN, "generations": generations,
    }))

    counts: dict[str, int] = {}
    for row in generations:
        counts[row["ending"]] = counts.get(row["ending"], 0) + 1
    log.info(f"wrote {len(generations)} generations to {args.out}: {counts}"
             + (f", {dropped} dropped" if dropped else ", none dropped"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path(".bak/evals"), help="shard-*/ from evals.py")
    parser.add_argument("--out", type=Path, default=Path(".bak/evalscope"))
    parser.add_argument("--data", type=Path, default=Path("."), help="where the stimulus files are")
    parser.add_argument("--pairs", type=Path, default=Path(".bak/probes/pairs.parquet"))
    parser.add_argument("--vectors", type=Path, default=Path(".bak/probes/diff.safetensors"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--neighbours", type=int, default=12)
    main(parser.parse_args())
