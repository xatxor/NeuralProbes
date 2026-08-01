#! /usr/bin/env python

"""Merge the dose shards and compute every statistic the report needs.

Three questions, in order.

**Q1 -- does a concept respond to the quantity?** For concept `c`, block `L` and ladder `l`, let
`d(t) = cos(c, h_top(t)) - cos(c, h_bottom(t))` be the change in the concept's value at token `t`
between the highest and lowest rung. The statistic is `max_t |d(t)|` with the argmax recorded, and
the 512 random directions and the `steps` control ladder are maximised over the same positions, so
the inflation that maximising introduces lands in the null rather than in the result.

The maximum is taken **only over tokens whose id is identical in every rung**. Taking it over all
positions instead puts the argmax on the varying digit for essentially every concept, because there
the two prompts hold different tokens and any direction separates their embeddings; run that way the
control ladder scores as high as the danger ladders (781 against 831 concepts at block 25) and the
statistic measures token identity rather than the model's assessment. The shared suffix is the
region where the prompts are character-for-character the same, so a difference there can only have
been carried in from the digit -- which is the comparison the original figure is about. The
differing region is reported separately as `local`, not mixed in.

**Q2 -- does it reach the reply?** Each rung was continued five times. The reply-side value of a
concept for one rung is the mean of its cosine over the generated tokens, averaged over the five
continuations. Per-token means rather than sums: replies grow from 88 to 131 tokens as the dose
rises, so any total would measure length.

**Q3 -- which prompt position predicts the reply?** Anthropic's section 2.2.2 reports r = 0.87 at the
Assistant colon against 0.59 on the user turn. The analogue here: for each concept, correlate its
prompt-side value at position `P` across the nine rungs against its reply-side value across the same
nine rungs, then compare positions by the distribution of |r| over concepts, with the random
directions as the floor.

Ragged ladders (`ibuprofen`) are aligned from the end. Their rungs share the template suffix exactly,
so negative indices refer to the same tokens across rungs even though absolute positions do not.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("doseall")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
CONTROL = "steps"
LADDERS = DANGER + (CONTROL,)
COSINE = 0

# Prompt positions compared in Q3. Negative indices count from the end of the prompt, which is the
# only indexing that means the same thing for aligned and ragged ladders alike.
POSITIONS = {
    "content mean": None,
    "assistant marker (-6)": -6,
    "newline after it (-5)": -5,
    "<think> (-4)": -4,
    "inside block (-3)": -3,
    "</think> (-2)": -2,
    "final token (-1)": -1,
}


def shards(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Load every shard and merge into one namespace.

    :param paths: shard npz files.

    :return: arrays keyed as written, the concatenated manifest, and the shared metadata.
    """
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for path in paths:
        blob = np.load(path, allow_pickle=False)
        meta = json.loads(str(blob["meta"]))
        manifest += json.loads(str(blob["manifest"]))
        for key in blob.files:
            if key not in ("manifest", "meta"):
                arrays[key] = blob[key]
    log.info(f"{len(paths)} shards, {len(manifest)} rungs, {len(arrays)} arrays")
    return arrays, manifest, meta


def ordered(manifest: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group rungs by condition, sorted by dose.

    :param manifest: the merged manifest.

    :return: `(ladder, rendering)` -> rungs in ascending dose order.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        groups[(row["ladder"], row["rendering"])].append(row)
    return {key: sorted(rows, key=lambda r: r["dose"]) for key, rows in groups.items()}


def suffix(arrays: dict[str, np.ndarray], rows: list[dict[str, Any]], ladder: str,
           render: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack the prompt readouts of one condition, aligned from the end, and mark the shared tokens.

    :param arrays: merged arrays.
    :param rows: rungs of this condition, dose-ordered.
    :param ladder: ladder name.
    :param render: rendering name.

    :return: `[rung, token, layer, column]` over the common tail, a boolean mask marking positions
        whose token is identical in every rung, and those token strings.
    """
    stack = [arrays[f"prompt.{ladder}.{render}.{row['rung']}"][COSINE] for row in rows]
    width = min(block.shape[0] for block in stack)
    grid = np.array([row["tokens"][-width:] for row in rows])
    return np.stack([block[-width:] for block in stack], axis=0), (grid == grid[0]).all(axis=0), list(grid[0])


def swing(values: np.ndarray, concepts: int, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Peak change between the extreme rungs over a chosen region, with the null measured the same way.

    :param values: `[rung, token, layer, column]`.
    :param concepts: number of real directions; the rest are controls.
    :param mask: `[token]`, which positions the maximum may be taken over.

    :return: `peak` and `argmax` `[layer, column]`, and `spread` `[layer]` from the controls. The
        argmax is an index into the full token axis, not into the masked subset.
    """
    delta = values[-1] - values[0]                       # [token, layer, column]
    scored = np.where(mask[:, None, None], np.abs(delta), -np.inf)
    index = scored.argmax(axis=0)                        # [layer, column]
    peak = np.take_along_axis(delta, index[None], axis=0)[0]
    spread = np.abs(peak[:, concepts:]).std(axis=-1)     # [layer], controls maximised identically
    return {"peak": peak, "argmax": index, "spread": spread}


def replies(arrays: dict[str, np.ndarray], rows: list[dict[str, Any]], ladder: str) -> np.ndarray:
    """Reply-side concept values, per-token mean, averaged over the five continuations.

    :param arrays: merged arrays.
    :param rows: rungs of this condition, dose-ordered.
    :param ladder: ladder name.

    :return: the five-continuation mean `[rung, layer, column]`, and each continuation separately
        `[rung, continuation, layer, column]` so a figure can carry the spread.
    """
    out, each = [], []
    for row in rows:
        per = [arrays[f"replymean.{ladder}.chat.{row['rung']}.{r['label']}"][COSINE]
               for r in row["replies"]]
        out.append(np.mean(per, axis=0) if per else np.zeros_like(out[-1]))
        each.append(np.stack(per, axis=0) if per else np.zeros_like(each[-1]))
    return np.stack(out, axis=0), np.stack(each, axis=0)


def spearman(values: np.ndarray) -> np.ndarray:
    """Rank correlation of every column against the rung order.

    :param values: `[rung, ...]`, rungs in ascending dose order.

    :return: correlation with `values.shape[1:]`.
    """
    order = values.argsort(axis=0)
    ranked = np.empty_like(values)
    grid = np.arange(values.shape[0]).reshape((-1,) + (1,) * (values.ndim - 1))
    np.put_along_axis(ranked, order, np.broadcast_to(grid, values.shape).astype(values.dtype), axis=0)
    dose = np.arange(values.shape[0], dtype=values.dtype)
    a = ranked - ranked.mean(0, keepdims=True)
    b = (dose - dose.mean()).reshape((-1,) + (1,) * (values.ndim - 1))
    denom = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), denom, out=np.zeros_like(denom), where=denom > 0)


def correlate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation along the first axis, column by column.

    :param left: `[n, ...]`.
    :param right: `[n, ...]`, same shape.

    :return: correlation with the trailing shape.
    """
    a = left - left.mean(0, keepdims=True)
    b = right - right.mean(0, keepdims=True)
    denom = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), denom, out=np.zeros_like(denom), where=denom > 0)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    arrays, manifest, meta = shards(sorted(args.root.glob("shard-*/dose-readout.npz")))
    groups = ordered(manifest)
    concepts, controls = meta["concepts"], meta["controls"]
    real = slice(0, concepts)
    pairs = pd.read_parquet(args.pairs)
    render = args.rendering

    # --- Q1: prompt side, maximum over token positions ------------------------------------------
    prompt: dict[str, dict[str, np.ndarray]] = {}
    tokens: dict[str, list[str]] = {}
    local: dict[str, dict[str, np.ndarray]] = {}
    for ladder in LADDERS:
        rows = groups[(ladder, render)]
        values, shared, names = suffix(arrays, rows, ladder, render)
        prompt[ladder] = swing(values, concepts, shared)
        local[ladder] = swing(values, concepts, ~shared)
        prompt[ladder]["values"] = values
        prompt[ladder]["shared"] = shared
        prompt[ladder]["rho"] = spearman(values)
        tokens[ladder] = names
        log.info(f"{ladder}: {values.shape} prompt readout, "
                 f"{int(shared.sum())} of {len(shared)} tokens identical across rungs")

    z_prompt = {l: prompt[l]["peak"] / np.maximum(prompt[l]["spread"][:, None], 1e-12) for l in LADDERS}
    z_local = {l: local[l]["peak"] / np.maximum(local[l]["spread"][:, None], 1e-12) for l in LADDERS}

    # Two fixed-position estimators to compare the maximum against. `final` reads the last prompt
    # token; `content` reads the mean over the user's sentence with every template token excluded.
    def fixed(values: np.ndarray) -> np.ndarray:
        delta = values[-1] - values[0]
        return delta / np.maximum(np.abs(delta[:, concepts:]).std(axis=-1, keepdims=True), 1e-12)

    z_final = {l: fixed(prompt[l]["values"][:, -1]) for l in LADDERS}
    z_content = {
        l: fixed(np.stack([arrays[f"mean.{l}.{render}.{r['rung']}"][COSINE]
                           for r in groups[(l, render)]]))
        for l in LADDERS
    }

    # An estimator is only useful if it says the same thing about the same concepts on two prompts
    # that differ only in wording. That is what this table measures, and it is what decides which
    # statistic the report is built on.
    def agreement(z: dict[str, np.ndarray], slot: int) -> dict[str, float]:
        return {f"{a}-{b}": float(np.corrcoef(z[a][slot, real], z[b][slot, real])[0, 1])
                for a, b in (("tylenol", "syrup"), ("tylenol", "ibuprofen"),
                             ("syrup", "ibuprofen"), ("tylenol", CONTROL))}

    estimators = {"max over identical tokens": z_prompt, "content mean": z_content,
                  "final token": z_final}

    # Every position, not only the three above. Two criteria are reported per position and block:
    # `dd`, the mean agreement among the three danger ladders, which measures whether the statistic
    # is the same function on differently worded prompts; and `dc`, its agreement with the magnitude
    # control, which should be low. They do not pick the same winner, and the winner by `dd - dc`
    # moves from block to block, so no position is selected on this evidence.
    width = min(prompt[l]["values"].shape[1] for l in LADDERS)
    positions = [("content mean", z_content), ("max", z_prompt)]
    positions += [(f"token {-k}", {l: fixed(prompt[l]["values"][:, -k]) for l in LADDERS})
                  for k in range(1, width + 1)]
    sweep: dict[str, dict[str, dict[str, float]]] = {}
    for name, z in positions:
        sweep[name] = {}
        for slot, layer in enumerate(LAYERS):
            dd = float(np.mean([np.corrcoef(z[a][slot, real], z[b][slot, real])[0, 1]
                                for a, b in (("tylenol", "syrup"), ("tylenol", "ibuprofen"),
                                             ("syrup", "ibuprofen"))]))
            dc = float(np.corrcoef(z["tylenol"][slot, real], z[CONTROL][slot, real])[0, 1])
            sweep[name][str(layer)] = {"dd": dd, "dc": dc, "separation": dd - dc}

    # --- Q2: reply side ------------------------------------------------------------------------
    reply, spread_reply = {}, {}
    for ladder in LADDERS:
        reply[ladder], spread_reply[ladder] = replies(arrays, groups[(ladder, "chat")], ladder)
    z_reply = {}
    for ladder in LADDERS:
        delta = reply[ladder][-1] - reply[ladder][0]
        z_reply[ladder] = delta / np.maximum(np.abs(delta[:, concepts:]).std(axis=-1)[:, None], 1e-12)

    # --- Q3: which prompt position predicts the reply -------------------------------------------
    prediction = {}
    for label, position in POSITIONS.items():
        rows = groups[("tylenol", render)]
        if position is None:
            side = np.stack([arrays[f"mean.tylenol.{render}.{r['rung']}"][COSINE] for r in rows])
        else:
            side = prompt["tylenol"]["values"][:, position]
        prediction[label] = correlate(side, reply["tylenol"])
    peak_side = np.stack([
        prompt["tylenol"]["values"][
            r, prompt["tylenol"]["argmax"], np.arange(len(LAYERS))[:, None],
            np.arange(concepts + controls)[None]]
        for r in range(prompt["tylenol"]["values"].shape[0])])
    prediction["peak token (per concept)"] = correlate(peak_side, reply["tylenol"])

    # --- table ----------------------------------------------------------------------------------
    rows = []
    for slot, layer in enumerate(LAYERS):
        for pair in range(concepts):
            row = pairs.iloc[pair]
            rows.append({
                "layer": layer, "pair": pair,
                "concept": row.get("concept", f"pair {pair}"),
                "antagonist": row.get("antagonist", ""), "class_name": row.get("class_name", ""),
                **{f"z_{l}": float(z_prompt[l][slot, pair]) for l in LADDERS},
                **{f"rz_{l}": float(z_reply[l][slot, pair]) for l in LADDERS},
                **{f"rho_{l}": float(prompt[l]["rho"][prompt[l]["argmax"][slot, pair], slot, pair])
                   for l in LADDERS},
                **{f"lz_{l}": float(z_local[l][slot, pair]) for l in LADDERS},
                **{f"f_{l}": float(z_final[l][slot, pair]) for l in LADDERS},
                **{f"c_{l}": float(z_content[l][slot, pair]) for l in LADDERS},
                "argmax": int(prompt["tylenol"]["argmax"][slot, pair]),
                "argmax_from_end": int(prompt["tylenol"]["argmax"][slot, pair]
                                       - prompt["tylenol"]["values"].shape[1]),
            })
    table = pd.DataFrame(rows)

    # Rows were built block-major, so a [layer, concept] array flattens onto them in C order.
    signs = np.stack([np.sign(z_final[l][:, real]) for l in DANGER])
    table["consistent"] = ((signs == signs[0]).all(axis=0) & (signs[0] != 0)).ravel()
    table["strength"] = np.min(
        np.stack([np.abs(z_final[l][:, real]) for l in DANGER]), axis=0).ravel()
    table["leak"] = table[f"f_{CONTROL}"].abs()
    table["specificity"] = table.strength - table.leak
    table["hit"] = table.consistent & (table.strength >= args.z)
    table.to_parquet(args.out / "dose-stats.parquet")

    hits = table[table.hit].sort_values("specificity", ascending=False)
    log.info(f"{len(hits)} concept-block cells clear |z|>={args.z} on all three danger ladders")

    # Everything the report quotes in prose, computed here so the document cannot go stale when the
    # vector set changes. The earlier draft had these as literals from the first run.
    def pc1(z: dict[str, np.ndarray], slot: int) -> float:
        stack = np.stack([z[l][slot, real] for l in DANGER], axis=1)
        singular = np.linalg.svd(stack - stack.mean(axis=0), compute_uv=False)
        return float(singular[0] ** 2 / (singular ** 2).sum())

    def modal(ladder: str, slot: int) -> tuple[int, int]:
        width = prompt[ladder]["values"].shape[1]
        values, counts_ = np.unique(prompt[ladder]["argmax"][slot, real], return_counts=True)
        best = int(np.argmax(counts_))
        return int(values[best]) - width, int(counts_[best])

    everything = {l: swing(prompt[l]["values"], concepts,
                           np.ones(prompt[l]["values"].shape[1], dtype=bool)) for l in LADDERS}
    z_every = {l: everything[l]["peak"] / np.maximum(everything[l]["spread"][:, None], 1e-12)
               for l in LADDERS}

    texts = {}
    for row in manifest:
        if row["ladder"] == "tylenol" and row["rendering"] == "chat":
            for entry in row["replies"]:
                if entry["label"] == "greedy":
                    texts[str(row["dose"])] = {"tokens": entry["tokens"], "text": entry["text"]}

    summary = {
        "meta": meta,
        "prose": {
            "alltoken_counts": {
                str(layer): {l: int((np.abs(z_every[l][slot, real]) >= args.z).sum()) for l in LADDERS}
                for slot, layer in enumerate(LAYERS)
            },
            "modal_peak": {l: {str(layer): modal(l, slot) for slot, layer in enumerate(LAYERS)}
                           for l in LADDERS},
            "pc1_prompt": {str(layer): pc1(z_final, slot) for slot, layer in enumerate(LAYERS)},
            "pc1_reply": {str(layer): pc1(z_reply, slot) for slot, layer in enumerate(LAYERS)},
            "reply_ladder_agreement": {
                str(layer): {f"tylenol-{l}": float(np.corrcoef(z_reply["tylenol"][slot, real],
                                                              z_reply[l][slot, real])[0, 1])
                             for l in ("syrup", "ibuprofen", CONTROL)}
                for slot, layer in enumerate(LAYERS)
            },
            "greedy": texts,
        },
        "rendering": render,
        "gate_z": args.z,
        "n_rungs": len(manifest),
        "n_replies": sum(len(r["replies"]) for r in manifest),
        "reply_tokens": {
            str(r["dose"]): [x["tokens"] for x in r["replies"]]
            for r in manifest if r["ladder"] == "tylenol" and r["rendering"] == "chat"
        },
        "shared_tokens": {l: [int(prompt[l]["shared"].sum()), int(len(prompt[l]["shared"]))]
                          for l in LADDERS},
        "sweep": sweep,
        "estimators": {
            name: {str(layer): agreement(z, slot) for slot, layer in enumerate(LAYERS)}
            for name, z in estimators.items()
        },
        "estimator_counts": {
            name: {str(layer): {l: int((np.abs(z[l][slot, real]) >= args.z).sum()) for l in LADDERS}
                   for slot, layer in enumerate(LAYERS)}
            for name, z in estimators.items()
        },
        "per_layer_final": {
            str(layer): {l: int((np.abs(z_final[l][slot, real]) >= args.z).sum()) for l in LADDERS}
            for slot, layer in enumerate(LAYERS)
        },
        "prompt_reply_agreement_final": {
            str(layer): float(np.corrcoef(z_final["tylenol"][slot, real],
                                          z_reply["tylenol"][slot, real])[0, 1])
            for slot, layer in enumerate(LAYERS)
        },
        "per_layer": {
            str(layer): {l: int((np.abs(z_prompt[l][slot, real]) >= args.z).sum()) for l in LADDERS}
            for slot, layer in enumerate(LAYERS)
        },
        "per_layer_local": {
            str(layer): {l: int((np.abs(z_local[l][slot, real]) >= args.z).sum()) for l in LADDERS}
            for slot, layer in enumerate(LAYERS)
        },
        "per_layer_reply": {
            str(layer): {l: int((np.abs(z_reply[l][slot, real]) >= args.z).sum()) for l in LADDERS}
            for slot, layer in enumerate(LAYERS)
        },
        "argmax_histogram": {
            str(layer): {
                str(int(k)): int(v) for k, v in
                zip(*np.unique(prompt["tylenol"]["argmax"][slot, real]
                               - prompt["tylenol"]["values"].shape[1], return_counts=True))
            } for slot, layer in enumerate(LAYERS)
        },
        "prediction": {
            label: {
                str(layer): {
                    "concepts_mean_abs_r": float(np.abs(values[slot, real]).mean()),
                    "controls_mean_abs_r": float(np.abs(values[slot, concepts:]).mean()),
                } for slot, layer in enumerate(LAYERS)
            } for label, values in prediction.items()
        },
        "prompt_reply_agreement": {
            str(layer): float(np.corrcoef(z_prompt["tylenol"][slot, real],
                                          z_reply["tylenol"][slot, real])[0, 1])
            for slot, layer in enumerate(LAYERS)
        },
        "tokens": tokens["tylenol"],
        "n_hits": int(len(hits)),
        "n_concepts": int(hits.pair.nunique()) if len(hits) else 0,
        "top": hits.head(args.top).to_dict("records"),
    }
    (args.out / "dose-summary.json").write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        args.out / "dose-derived.npz",
        **{f"zp.{l}": z_prompt[l] for l in LADDERS},
        **{f"zr.{l}": z_reply[l] for l in LADDERS},
        **{f"pv.{l}": prompt[l]["values"] for l in LADDERS},
        **{f"rv.{l}": reply[l] for l in LADDERS},
        **{f"rs.{l}": spread_reply[l] for l in LADDERS},
        **{f"am.{l}": prompt[l]["argmax"] for l in LADDERS},
        **{f"zl.{l}": z_local[l] for l in LADDERS},
        **{f"zf.{l}": z_final[l] for l in LADDERS},
        **{f"zc.{l}": z_content[l] for l in LADDERS},
        **{f"sh.{l}": prompt[l]["shared"] for l in LADDERS},
        **{f"pr.{i}": v for i, v in enumerate(prediction.values())},
        prednames=np.array(json.dumps(list(prediction))),
        tokens=np.array(json.dumps(tokens["tylenol"])),
        doses=np.array(json.dumps({l: [r["dose"] for r in groups[(l, render)]] for l in LADDERS})),
    )
    log.info(f"wrote dose-stats.parquet, dose-summary.json, dose-derived.npz into {args.out}")

    print(f"\nrungs {len(manifest)}  replies {summary['n_replies']}  hits {len(hits)} "
          f"({summary['n_concepts']} concepts)")
    print("\nposition sweep, block 25, best five by separation (dd - dc):")
    for name, row in sorted(sweep.items(), key=lambda kv: -kv[1]["25"]["separation"])[:5]:
        r = row["25"]
        print(f"  {name:<14} dd {r['dd']:+.3f}  dc {r['dc']:+.3f}  separation {r['separation']:+.3f}")
    print("\n  best by separation per block: " + ", ".join(
        f"L{layer} {max(sweep, key=lambda n: sweep[n][str(layer)]['separation'])}"
        for layer in LAYERS))
    print("\n  highest danger-danger agreement per block: " + ", ".join(
        f"L{layer} {max(sweep, key=lambda n: sweep[n][str(layer)]['dd'])}" for layer in LAYERS))
    print("\nestimator agreement at block 25 (danger-danger should be high, danger-control low):")
    for name in estimators:
        print(f"  {name:<26} " + "  ".join(f"{k} {v:+.3f}"
                                           for k, v in summary["estimators"][name]["25"].items()))
    print("\nprompt-side |z|>=%.0f per block, final token:" % args.z)
    for layer in LAYERS:
        print(f"  block {layer}: {summary['per_layer_final'][str(layer)]}")
    print("\nlocal (differing tokens only) |z|>=%.0f per block:" % args.z)
    for layer in LAYERS:
        print(f"  block {layer}: {summary['per_layer_local'][str(layer)]}")
    print("\nreply-side |z|>=%.0f per block:" % args.z)
    for layer in LAYERS:
        print(f"  block {layer}: {summary['per_layer_reply'][str(layer)]}")
    print("\nQ3, mean |r| between prompt position and reply content (block 25):")
    for label in prediction:
        row = summary["prediction"][label]["25"]
        print(f"  {label:<28} concepts {row['concepts_mean_abs_r']:.3f}   "
              f"random {row['controls_mean_abs_r']:.3f}")
    print("\ntop 15 by specificity:")
    print(hits.head(15)[["layer", "concept", "z_tylenol", "z_syrup", "z_ibuprofen", "z_steps",
                         "rz_tylenol", "argmax_from_end"]].to_string(index=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pairs", type=Path, default=Path("pairs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--rendering", default="chat", choices=["chat", "raw"])
    parser.add_argument("--z", type=float, default=4.0)
    parser.add_argument("--top", type=int, default=40)
    main(parser.parse_args())
