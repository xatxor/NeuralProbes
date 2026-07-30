#! /usr/bin/env python

"""Does the answer depend on which token we read?

The paper reads at what it calls the Assistant colon, defining it as "the ':' token after
'Assistant', the last token before the Assistant's response". Qwen3 has no such colon. With
reasoning disabled its prompt ends `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`, so the last
token before generation is `\\n\\n`. That is the same *position* under their definition and a
different token, and the substitution was made without checking whether it matters.

This checks. Two comparisons:

**Across positions within the chat rendering.** The same statistic is recomputed at the digit
itself, at the end of the user's sentence, at `<|im_end|>`, at the `assistant` marker, at `</think>`
and at the final token. If the result is a property of the model's situational assessment it should
survive moving a few tokens; if it lives on one special token it is a property of that token.

**Against the raw rendering.** `Human: ... \\n\\nAssistant:` ends in a literal ':' and uses no special
tokens at all -- which is both the paper's own format and the format our vectors were extracted
under. If the two renderings disagree, the chat-template result is a formatting artefact.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("dosewhere")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
COSINE = 0


def zscores(blob: Any, key: str, position: int, concepts: int) -> np.ndarray:
    """Swing z-score at one token position, for every direction.

    :param blob: the loaded npz.
    :param key: `ladder.rendering`.
    :param position: token index, negative counts from the end.
    :param concepts: how many leading columns are real concepts.

    :return: `[layer, column]` z-scores against the random-direction spread.
    """
    values = blob[f"{key}.values"][:, COSINE]
    swing = values[-1, position] - values[0, position]
    control = swing[:, concepts:]
    return (swing - control.mean(-1, keepdims=True)) / np.maximum(control.std(-1, keepdims=True), 1e-12)


def summarise(blob: Any, render: str, position: int, concepts: int, gate: float) -> dict[str, Any]:
    """One row of the comparison: how many concepts respond, and how much the ladders agree.

    :param blob: the loaded npz.
    :param render: `chat` or `raw`.
    :param position: token index to read at.
    :param concepts: number of real directions.
    :param gate: the |z| threshold counted as a response.

    :return: counts, medians and the first-principal-component share across the danger ladders.
    """
    z = {l: zscores(blob, f"{l}.{render}", position if l != "ibuprofen" else -1, concepts)
         for l in DANGER + ("steps",)}
    slot = LAYERS.index(25)
    stack = np.stack([z[l][slot, :concepts] for l in DANGER], axis=1)
    centred = stack - stack.mean(axis=0)
    share = float(np.linalg.svd(centred, compute_uv=False)[0] ** 2 / (np.linalg.svd(centred, compute_uv=False) ** 2).sum())
    return {
        "n_tylenol": int((np.abs(z["tylenol"][slot, :concepts]) >= gate).sum()),
        "n_steps": int((np.abs(z["steps"][slot, :concepts]) >= gate).sum()),
        "median_z": float(np.median(np.abs(z["tylenol"][slot, :concepts]))),
        "max_z": float(np.abs(z["tylenol"][slot, :concepts]).max()),
        "pc1_share": share,
        "z": z,
    }


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    blob = np.load(args.readout, allow_pickle=False)
    manifest = {f"{e['ladder']}.{e['rendering']}": e for e in manifest_of(blob)}
    meta = json.loads(str(blob["meta"]))
    concepts = meta["concepts"]
    pairs = pd.read_parquet(args.pairs)

    chat_tokens = manifest["tylenol.chat"]["tokens"]
    raw_tokens = manifest["tylenol.raw"]["tokens"]
    digit = manifest["tylenol.chat"]["varying"][0]

    # Labelled by what the token actually is, verified against the printed token column rather than
    # counted back from the end by hand -- the first version of this table had three rows mislabelled
    # exactly that way.
    positions = [
        ("the digit itself", digit),
        ("end of user sentence", -8),
        ("assistant marker", -6),
        ("newline after it", -5),
        ("<think>", -4),
        ("inside the block", -3),
        ("</think>", -2),
        ("final token (used)", -1),
    ]

    print(f"chat rendering ends: {[t for t in chat_tokens[-6:]]}")
    print(f"raw  rendering ends: {[t for t in raw_tokens[-4:]]}")
    print()
    print(f"{'position':<24} {'token':<14} {'|z|>=' + str(args.gate):>8} {'steps':>7} "
          f"{'median':>7} {'max':>7} {'PC1':>7}")
    print("-" * 80)
    for label, position in positions:
        row = summarise(blob, "chat", position, concepts, args.gate)
        token = chat_tokens[position] if position >= 0 else chat_tokens[len(chat_tokens) + position]
        print(f"{label:<24} {token!r:<14} {row['n_tylenol']:>8} {row['n_steps']:>7} "
              f"{row['median_z']:>7.2f} {row['max_z']:>7.2f} {row['pc1_share']:>6.1%}")

    print()
    raw = summarise(blob, "raw", -1, concepts, args.gate)
    chat = summarise(blob, "chat", -1, concepts, args.gate)
    print(f"{'raw, final token (:)':<24} {raw_tokens[-1]!r:<14} {raw['n_tylenol']:>8} "
          f"{raw['n_steps']:>7} {raw['median_z']:>7.2f} {raw['max_z']:>7.2f} {raw['pc1_share']:>6.1%}")

    slot = LAYERS.index(25)
    agreement = np.corrcoef(chat["z"]["tylenol"][slot, :concepts], raw["z"]["tylenol"][slot, :concepts])[0, 1]
    print()
    print(f"agreement between chat and raw on which concepts respond: pearson r = {agreement:.3f}")

    print()
    print("top 12 by |z| in the RAW rendering (paper's own format, literal ':' token):")
    order = np.argsort(-np.abs(raw["z"]["tylenol"][slot, :concepts]))[:12]
    for rank, pair in enumerate(order, 1):
        print(f"  {rank:2d}. {str(pairs.iloc[pair].get('concept', pair))[:44]:<44} "
              f"z={raw['z']['tylenol'][slot, pair]:6.1f}  steps={raw['z']['steps'][slot, pair]:6.1f}")


def manifest_of(blob: Any) -> list[dict[str, Any]]:
    """Pull the manifest out of the npz.

    :param blob: the loaded npz.

    :return: one dict per condition.
    """
    return json.loads(str(blob["manifest"]))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, default=Path("dose-readout.npz"))
    parser.add_argument("--pairs", type=Path, default=Path("pairs.parquet"))
    parser.add_argument("--gate", type=float, default=4.0)
    main(parser.parse_args())
