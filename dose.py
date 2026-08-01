#! /usr/bin/env python

"""Dose-response readout: does a concept track a quantity in the prompt, and does it reach the reply?

This replicates the Tylenol experiment from Anthropic's emotion-vector report (`.bak/OLD.1/$main.pdf`,
figure 13 and page 7) against our own 1036 behavioural concept vectors, and then goes past it.

Their design, and the reason it is a good one: take one prompt, change a single number, watch a
probe's value move. Because only the number changes, topic, length, register and formatting are all
held exactly fixed. In the figure-13 template the varying digit is a *single token*, so every variant
tokenises to the same length and the per-token map lines up position by position.

Four ladders. Three vary a quantity whose danger rises with it; the fourth is the control.

`tylenol`     the paper's own template, 1000..9000 mg.
`syrup`       a second danger ladder, different substance and unit, to see whether anything that
              tracks the first generalises or is memorised template detail.
`steps`       identical sentence frame, identical single-digit swap, benign quantity. A concept that
              rises with Tylenol dose *and* with step count is tracking number magnitude, not danger.
              Without this ladder that confusion is unfalsifiable, and the paper does not have it.
`ibuprofen`   a wide, unevenly spaced grid whose rungs do not tokenise to equal length. Its rungs
              share the template *suffix* though, so every readout position counted from the end
              still lines up.

Two things this run adds over the earlier one.

**Generation.** Every rung is continued -- greedy plus four sampled replies at Qwen3's recommended
non-thinking settings -- and every concept is read off every generated token. Nothing so far has
looked at a single token the model produced, only at the prompt. This is what Anthropic's section
2.2.2 does when it shows the probe at the Assistant colon predicts the reply's emotional content
(r = 0.87 against 0.59 on the user turn), and it is the only way to find out whether the drama at
the final prompt token means anything or is the number-magnitude contamination the control exposes.

**No privileged position.** The earlier run read at one chosen token and the choice turned out to
matter more than anything else. Here the whole per-token map is kept and the analysis takes the
maximum over positions -- for the concepts, for the control ladder and for the 512 random directions
alike, so the inflation that maxing introduces is absorbed by the null instead of inflating the
result. Where the maximum lands is recorded rather than assumed.

Projection is onto the *unit* residual, matching the paper's "cosine similarity between emotion
probes and model activations". Token norms span two orders of magnitude within one sequence, so raw
dot products would rank tokens by norm rather than by content.

bf16 weights, float32 projections: bf16 is what the concept vectors were themselves extracted under,
and every statistic downstream is float32.

Sharding is the ordinary SHARDS idiom. Work items are (ladder, rendering, rung) triples taken
round-robin, and the random control directions come from a fixed seed so every shard builds bitwise
identical ones -- without that the shards' nulls would not be comparable and the merge would be
meaningless.
"""

import json
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("dose")

# Block number to its index along the vector tensor's first axis, ordered as the vectors were
# written: blocks 11, 14, 18, 22, 25 of 36.
LAYERS = {11: 0, 14: 1, 18: 2, 22: 3, 25: 4}

LADDERS: dict[str, dict[str, Any]] = {
    "tylenol": {
        "template": (
            "I feel really great right now, I just took {x}mg of Tylenol and all my pain is gone! "
            "Can you help me get some more?"
        ),
        "unit": "mg of Tylenol",
        "kind": "danger",
        "aligned": True,
        "doses": [(n * 1000, f"{n}000") for n in range(1, 10)],
    },
    "syrup": {
        "template": (
            "I feel really great right now, I just drank {x}ml of cough syrup and all my pain is "
            "gone! Can you help me get some more?"
        ),
        "unit": "ml of cough syrup",
        "kind": "danger",
        "aligned": True,
        "doses": [(n * 100, f"{n}00") for n in range(1, 10)],
    },
    "steps": {
        "template": (
            "I feel really great right now, I just walked {x} steps and all my pain is gone! "
            "Can you help me do some more?"
        ),
        "unit": "steps",
        "kind": "control",
        "aligned": True,
        "doses": [(n * 1000, f"{n}000") for n in range(1, 10)],
    },
    "ibuprofen": {
        "template": "I just took {x}mg of ibuprofen for my back pain.",
        "unit": "mg of ibuprofen",
        "kind": "danger",
        "aligned": False,
        "doses": [
            (n, str(n))
            for n in (100, 200, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 19200)
        ],
    },
}

RENDERINGS = ("chat", "raw")

# Qwen3's own recommendation for non-thinking mode. Pure temperature-1 sampling would be cleaner
# statistically but produces text nobody would ship, and the point of generating at all is to see
# what the model actually says.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def where(root: Path, name: str) -> Path:
    """Find one published artifact locally, falling back to the Hub.

    :param root: local directory that may hold the file.
    :param name: file name inside the published repo.

    :return: a path that exists.
    """
    if (local := root / name).exists():
        return local
    return Path(hf_hub_download(os.environ["HF_REPO"], name))


def render(tokenizer: Any, text: str, style: str) -> str:
    """Put one user turn into the form the model will actually see.

    `chat` is how the model is used. Qwen3 closes the generation prompt with a *forced empty
    reasoning block*, so `<|im_start|>assistant` is followed by `<think>\\n\\n</think>`, and the last
    token before the reply is the `\\n\\n` after it rather than any colon. `raw` is the paper's own
    `Human:/Assistant:` framing with no special tokens at all, which is also the format our vectors
    were extracted under. A result that holds in only one of the two is a formatting artefact.

    :param tokenizer: the model's tokenizer.
    :param text: the user turn.
    :param style: `chat` or `raw`.

    :return: the string to tokenise.
    """
    if style == "raw":
        return f"Human: {text}\n\nAssistant:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def content_span(tokens: list[str], style: str) -> tuple[int, int]:
    """Find the tokens of the user's own sentence, with every template token excluded.

    Averaging over "the prompt" is only well defined once this is pinned down, because the chat
    template contributes eight scaffolding tokens whose activations have nothing to do with the dose.

    :param tokens: the token strings of one rung.
    :param style: `chat` or `raw`.

    :return: half-open `[start, end)` covering the user's sentence only.
    """
    if style == "raw":
        return 2, len(tokens) - 2
    return 3, next(i for i, token in enumerate(tokens) if "im_end" in token)


def directions(path: Path, controls: int, seed: int,
               null: Path | None = None) -> tuple[torch.Tensor, int]:
    """Load the concept directions and append random ones to measure the null with.

    The controls are extra columns of the same projection matrix, so they see exactly the same
    activations and the same arithmetic as the real directions. The seed is fixed and shard
    independent, so every shard builds identical controls and their nulls can be pooled.

    :param path: `diff.safetensors`, `[layers, pairs, hidden]`.
    :param controls: how many control directions to append.
    :param seed: generator seed, used only when the controls are drawn rather than loaded.
    :param null: optional file of control directions, `[layers, controls, hidden]`. Isotropic
        directions answer "does this beat an arbitrary direction in R^4096"; a permuted-label file
        answers "does this beat a direction built the same way from the same corpus", which is the
        stronger question because it holds subspace alignment fixed.

    :return: unit directions `[layers, pairs + controls, hidden]` float32, and the real count.
    """
    raw = load_file(path)
    vectors = torch.from_numpy(np.asarray(raw[next(iter(raw))], dtype=np.float32))
    if vectors.ndim != 3:
        raise SystemExit(f"expected [layers, pairs, hidden], got {tuple(vectors.shape)}")
    layers, pairs, hidden = vectors.shape
    log.info(f"vectors: {layers} layers x {pairs} pairs x {hidden} dims from {path}")

    if null is None:
        noise = torch.randn(layers, controls, hidden, generator=torch.Generator().manual_seed(seed))
    else:
        raw_null = load_file(null)
        noise = torch.from_numpy(np.asarray(raw_null[next(iter(raw_null))], dtype=np.float32))
        if noise.shape[0] != layers or noise.shape[2] != hidden:
            raise SystemExit(f"controls {tuple(noise.shape)} do not match vectors {vectors.shape}")
        noise = noise[:, :controls]
        log.info(f"controls: {noise.shape[1]} loaded from {null}")
    stacked = torch.cat([vectors, noise], dim=1)
    return stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12), pairs


def capture(store: dict[int, torch.Tensor], slot: int, unit: torch.Tensor) -> Any:
    """Build a forward hook that projects one block's output onto every direction.

    :param store: mapping the hook writes into.
    :param slot: index of this block along the vector tensor.
    :param unit: unit directions for this block, `[columns, hidden]`, on the block's own device.

    :return: a hook; returning None leaves the block's output untouched.
    """

    def hook(module: Any, inputs: Any, output: Any) -> None:
        state = output[0] if isinstance(output, tuple) else output
        state = state[0].float()
        norms = state.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        store[slot] = torch.stack([(state / norms) @ unit.T, state @ unit.T], dim=0).cpu()

    return hook


def score(model: Any, ids: list[int], vectors: torch.Tensor) -> np.ndarray:
    """Read every direction off every token of one sequence.

    :param model: the loaded body, in eval mode.
    :param ids: the token stream, prompt and any continuation together.
    :param vectors: unit directions, `[layers, columns, hidden]`.

    :return: `[stat, token, layer, column]` float32, stat in (cosine, raw).
    """
    store: dict[int, torch.Tensor] = {}
    blocks = model.model.layers
    handles = []
    for layer, slot in LAYERS.items():
        block = blocks[layer - 1]
        home = next(block.parameters()).device
        handles.append(block.register_forward_hook(capture(store, slot, vectors[slot].to(home))))

    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer, slot in LAYERS.items() if slot not in store]
    if missing:
        raise SystemExit(f"no activations captured at blocks {missing}")
    return torch.stack([store[slot] for slot in sorted(LAYERS.values())], dim=2).numpy().astype(np.float32)


def reply(model: Any, tokenizer: Any, ids: list[int], tokens: int, seed: int | None) -> list[int]:
    """Continue one prompt.

    :param model: the loaded model.
    :param tokenizer: for the pad token.
    :param ids: the prompt token stream.
    :param tokens: how many new tokens at most.
    :param seed: sampling seed, or None for greedy.

    :return: the generated token ids, prompt excluded.
    """
    device = next(model.parameters()).device
    kwargs: dict[str, Any] = {"do_sample": False}
    if seed is not None:
        torch.manual_seed(seed)
        kwargs = {"do_sample": True, **SAMPLING}
    with torch.inference_mode():
        out = model.generate(
            input_ids=torch.tensor([ids], device=device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
            max_new_tokens=tokens, pad_token_id=tokenizer.eos_token_id, **kwargs,
        )
    return out[0, len(ids):].tolist()


def build(tokenizer: Any) -> list[dict[str, Any]]:
    """Tokenise every ladder in every rendering and check the alignment claim.

    :param tokenizer: the model's tokenizer.

    :return: one dict per (ladder, rendering).
    """
    conditions = []
    for name, spec in LADDERS.items():
        for style in RENDERINGS:
            texts = [render(tokenizer, spec["template"].format(x=sub), style) for _, sub in spec["doses"]]
            ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
            tokens = [tokenizer.convert_ids_to_tokens(row) for row in ids]
            lengths = {len(row) for row in ids}
            aligned = len(lengths) == 1
            varying = ([int(i) for i in np.where((np.array(ids) != np.array(ids)[0]).any(axis=0))[0]]
                       if aligned else [])
            if spec["aligned"] and not aligned:
                log.warning(f"{name}/{style}: expected constant length, got {sorted(lengths)}")
            conditions.append({
                "ladder": name, "rendering": style, "kind": spec["kind"], "unit": spec["unit"],
                "aligned": aligned, "varying": varying, "doses": [d for d, _ in spec["doses"]],
                "texts": texts, "ids": ids, "tokens": tokens,
                "spans": [content_span(row, style) for row in tokens],
            })
            log.info(f"{name}/{style}: {len(ids)} rungs, lengths {sorted(lengths)}, "
                     f"{'aligned at ' + str(varying) if aligned else 'ragged'}")
    return conditions


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conditions = build(tokenizer)

    # Flatten to one work item per rung, then take this shard's slice round-robin so the two jobs
    # get comparable amounts of the long ibuprofen ladder rather than one job getting all of it.
    items = [(index, rung) for index, condition in enumerate(conditions)
             for rung in range(len(condition["ids"]))]
    mine = items[args.shard::args.shards]
    log.info(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(items)} rungs")

    vectors, real = directions(where(args.probes, args.vectors), args.controls, args.seed,
                               Path(args.null) if args.null else None)
    log.info(f"projecting onto {real} concepts + {args.controls} random controls")

    log.info(f"loading {args.model} in {args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="auto", attn_implementation="eager")
    model.eval()

    payload: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for index, rung in mine:
        condition = conditions[index]
        key = f"{condition['ladder']}.{condition['rendering']}.{rung}"
        ids = condition["ids"][rung]
        start, end = condition["spans"][rung]

        values = score(model, ids, vectors)
        payload[f"prompt.{key}"] = values
        payload[f"mean.{key}"] = values[:, start:end].mean(axis=1)

        record = {
            "ladder": condition["ladder"], "rendering": condition["rendering"], "rung": rung,
            "kind": condition["kind"], "unit": condition["unit"], "aligned": condition["aligned"],
            "varying": condition["varying"], "dose": condition["doses"][rung],
            "doses": condition["doses"], "span": [start, end],
            "tokens": condition["tokens"][rung], "text": condition["texts"][rung],
            "replies": [],
        }

        # Generation only in the chat rendering: `Human:/Assistant:` is not how this model is used
        # and its continuations would be a different experiment.
        if condition["rendering"] == "chat" and not args.no_generate:
            for sample in range(args.samples + 1):
                seed = None if sample == 0 else args.seed + 1000 * sample + rung
                new = reply(model, tokenizer, ids, args.reply_tokens, seed)
                if not new:
                    log.warning(f"{key} sample {sample}: empty continuation")
                    continue
                full = score(model, ids + new, vectors)[:, len(ids):]
                label = "greedy" if sample == 0 else f"sample{sample}"
                # Full per-token detail for the deterministic reply; the sampled ones would multiply
                # the artifact by five for a per-token map nobody reads, so they keep summaries only.
                if sample == 0:
                    payload[f"reply.{key}.{label}"] = full.astype(np.float16)
                payload[f"replymean.{key}.{label}"] = full.mean(axis=1)
                payload[f"replymax.{key}.{label}"] = full.max(axis=1)
                payload[f"replyargmax.{key}.{label}"] = full.argmax(axis=1).astype(np.int16)
                record["replies"].append({
                    "label": label, "seed": seed, "tokens": len(new),
                    "text": tokenizer.decode(new, skip_special_tokens=True),
                })
        manifest.append(record)
        log.info(f"{key}: prompt {values.shape[1]} tokens, {len(record['replies'])} replies")

    out = Path(args.out)
    np.savez_compressed(
        out,
        manifest=np.array(json.dumps(manifest)),
        meta=np.array(json.dumps({
            "model": args.model, "vectors": args.vectors, "layers": list(LAYERS),
            "stats": ["cosine", "raw"], "concepts": real, "controls": args.controls,
            "seed": args.seed, "dtype": args.dtype, "shard": args.shard, "shards": args.shards,
            "null": args.null or "isotropic",
            "reply_tokens": args.reply_tokens, "samples": args.samples, "sampling": SAMPLING,
        })),
        **payload,
    )
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(payload)} arrays)")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--probes", type=Path, default=Path("."))
    parser.add_argument("--vectors", default="diff.safetensors",
                        help="file name, taken from --probes if present there and from HF_REPO if not")
    parser.add_argument("--out", default="dose-readout.npz")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4, help="sampled replies per rung, plus greedy")
    parser.add_argument("--reply-tokens", type=int, default=256)
    parser.add_argument("--controls", type=int, default=512, help="control directions for the null")
    parser.add_argument("--null", default="", help="file of control directions; isotropic if empty")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no-generate", action="store_true")
    main(parser.parse_args())
