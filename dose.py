#! /usr/bin/env python

"""Dose-response readout: does a concept's value track a quantity buried in the prompt?

This replicates the Tylenol experiment from Anthropic's emotion-vector report (`.bak/OLD.1/$main.pdf`,
figure 13 and the bullet list on page 7) against our own 1036 behavioural concept vectors.

Their design, and the reason it is a good one: take one prompt, change a single number in it, and
watch a probe's value move. Because only the number changes, everything a probe might otherwise be
picking up -- topic, length, register, formatting -- is held exactly fixed. In the figure-13 template
the varying digit is a *single token*, so every variant tokenises to the same length and the whole
per-token map lines up position by position. That is what lets you say "the difference appears at
this token, in these layers" rather than only "the two prompts differ somewhere".

Four ladders run here, and the last two are the point of the exercise rather than decoration:

`tylenol`     the paper's own template, 1000..9000 mg. Danger rises with the number.
`syrup`       a second danger ladder, different substance and unit, to see whether anything that
              tracks the first generalises or is memorised template detail.
`steps`       identical sentence frame, identical single-digit swap, but the quantity is benign.
              A concept that rises with Tylenol dose *and* with step count is tracking number
              magnitude, not danger. Without this ladder that confusion is unfalsifiable.
`ibuprofen`   a wide, unevenly spaced grid whose variants do *not* tokenise to equal length. Only
              the final token is comparable across it, which is fine because that is the token the
              paper reports anyway, and 14 rungs buy far more statistical power than 9.

Two nulls, because a correlation over nine points is easy to obtain by accident. 512 random unit
directions ride along in the same projection and give an empirical distribution of |rho| under the
null of no relationship; the `steps` ladder gives the stronger, semantic null described above.

Projection is onto the *unit* residual, matching `highlight.py` and the paper's own "cosine
similarity between emotion probes and model activations". Token norms span two orders of magnitude
within a sequence, so raw dot products would rank tokens by norm rather than by content. The raw
projection is stored too, since across a fixed position the norm barely moves and the difference is
worth being able to check rather than assert.

float32 throughout. The cards are V100 (sm_70), which has no bf16 at all, and there is no reason to
economise: the entire experiment is roughly eighty forward passes over fifty tokens.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("dose")

# Block number to its index along the vector tensor's first axis, ordered as the vectors were
# written: blocks 11, 14, 18, 22, 25 of 36.
LAYERS = {11: 0, 14: 1, 18: 2, 22: 3, 25: 4}

# Each ladder is (template, unit, [(dose, substitution)]). `aligned` ladders vary a single digit and
# so tokenise to a constant length; that is asserted at build time, never assumed.
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

# How much of the shared template suffix to keep for ladders whose rungs tokenise to different
# lengths. Ten covers everything from `<|im_end|>` to the final token.
TAIL = 10


def render(tokenizer: Any, text: str, style: str) -> str:
    """Put one user turn into the form the model will actually see.

    Two styles are kept because they answer different questions. `chat` is how the model is used and
    is where the interesting token lives -- Qwen3 closes the generation prompt with a *forced empty
    reasoning block*, so `<|im_start|>assistant` is followed by `<think>\\n\\n</think>`, and the
    analogue of the paper's "Assistant:" colon is the last token of that. `raw` is the paper's own
    `Human:/Assistant:` framing with no special tokens at all, which is also the format our concept
    vectors were extracted under. If a result holds only in one of the two it is a formatting
    artefact and should be reported as one.

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


def directions(path: Path, controls: int, seed: int) -> tuple[torch.Tensor, int]:
    """Load the concept directions and append random ones to measure the null with.

    The controls are not a separate pass. They are extra columns of the same projection matrix, so
    they see exactly the same activations, the same tokens and the same arithmetic as the real
    directions -- which is the only way the resulting null is comparable.

    :param path: `diff.safetensors`, `[layers, pairs, hidden]`.
    :param controls: how many random unit directions to append.
    :param seed: generator seed for those directions.

    :return: unit directions `[layers, pairs + controls, hidden]` in float32, and the real count.
    """
    raw = load_file(path)
    key = next(iter(raw))
    vectors = torch.from_numpy(np.asarray(raw[key], dtype=np.float32))
    if vectors.ndim != 3:
        raise SystemExit(f"expected [layers, pairs, hidden], got {tuple(vectors.shape)}")
    layers, pairs, hidden = vectors.shape
    log.info(f"vectors: {layers} layers x {pairs} pairs x {hidden} dims from {path}")

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(layers, controls, hidden, generator=generator)
    stacked = torch.cat([vectors, noise], dim=1)
    stacked = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return stacked, pairs


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
        cosine = (state / norms) @ unit.T
        store[slot] = torch.stack([cosine, state @ unit.T], dim=0).cpu()
        store[-1 - slot] = norms[:, 0].cpu()

    return hook


def score(model: Any, ids: list[int], vectors: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Read every direction off every token of one sequence.

    :param model: the loaded body, in eval mode.
    :param ids: the token stream.
    :param vectors: unit directions, `[layers, columns, hidden]`.

    :return: `[stat, token, layer, column]` float32 with stat in (cosine, raw), and the per-layer
        residual norm `[token, layer]`.
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

    values = torch.stack([store[slot] for slot in sorted(LAYERS.values())], dim=2)
    norms = torch.stack([store[-1 - slot] for slot in sorted(LAYERS.values())], dim=1)
    return values.numpy().astype(np.float32), norms.numpy().astype(np.float32)


def content_span(tokens: list[str], style: str) -> tuple[int, int]:
    """Find the tokens of the user's own sentence, with every template token excluded.

    Averaging over "the prompt" is only well defined once this is pinned down, because the chat
    template contributes eight scaffolding tokens whose activations have nothing to do with the dose.
    Under Qwen3 the layout is `<|im_start|> user \\n {content} <|im_end|> \\n <|im_start|> assistant
    \\n <think> \\n\\n </think> \\n\\n`, so the content is everything between the third token and the
    `<|im_end|>`.

    :param tokens: the token strings of one rung.
    :param style: `chat` or `raw`.

    :return: half-open `[start, end)` covering the user's sentence only.
    """
    if style == "raw":
        # `Human: {text}\n\nAssistant:` -- drop the two-token header and the two-token footer.
        return 2, len(tokens) - 2
    end = next(i for i, token in enumerate(tokens) if "im_end" in token)
    return 3, end


def build(tokenizer: Any) -> list[dict[str, Any]]:
    """Tokenise every ladder in every rendering, and check the alignment claim.

    A ladder marked `aligned` promises that all its rungs have the same token count and differ at
    exactly one position. That is the whole basis of the per-token comparison, so it is verified
    here and the differing positions are recorded rather than trusted.

    :param tokenizer: the model's tokenizer.

    :return: one dict per (ladder, rendering) with ids, token strings, doses and the diff positions.
    """
    conditions = []
    for name, spec in LADDERS.items():
        for style in RENDERINGS:
            texts = [render(tokenizer, spec["template"].format(x=sub), style) for _, sub in spec["doses"]]
            ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
            lengths = {len(row) for row in ids}
            aligned = len(lengths) == 1

            varying: list[int] = []
            if aligned:
                grid = np.array(ids)
                varying = [int(i) for i in np.where((grid != grid[0]).any(axis=0))[0]]

            if spec["aligned"] and not aligned:
                log.warning(f"{name}/{style}: expected constant length, got {sorted(lengths)}")

            conditions.append(
                {
                    "ladder": name,
                    "rendering": style,
                    "kind": spec["kind"],
                    "unit": spec["unit"],
                    "aligned": aligned,
                    "varying": varying,
                    "doses": [d for d, _ in spec["doses"]],
                    "texts": texts,
                    "ids": ids,
                    "tokens": [tokenizer.convert_ids_to_tokens(row) for row in ids],
                    "spans": [
                        content_span(tokenizer.convert_ids_to_tokens(row), style) for row in ids
                    ],
                }
            )
            log.info(
                f"{name}/{style}: {len(ids)} rungs, lengths {sorted(lengths)}, "
                f"{'aligned at ' + str(varying) if aligned else 'ragged'}"
            )
    return conditions


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conditions = build(tokenizer)

    vectors, real = directions(Path(args.vectors), args.controls, args.seed)
    log.info(f"projecting onto {real} concepts + {args.controls} random controls")

    log.info(f"loading {args.model} in float32 across the visible cards")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    payload: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for condition in conditions:
        stack, norms, means = [], [], []
        for ids, (start, end) in zip(condition["ids"], condition["spans"]):
            values, layer_norms = score(model, ids, vectors)
            stack.append(values)
            norms.append(layer_norms)
            # The mean over the user's own sentence, which is defined even when the rungs are ragged
            # and so is the one position comparable across every ladder.
            means.append(values[:, start:end].mean(axis=1))
        key = f"{condition['ladder']}.{condition['rendering']}"
        payload[f"{key}.mean"] = np.stack(means, axis=0)
        if condition["aligned"]:
            payload[f"{key}.values"] = np.stack(stack, axis=0)
            payload[f"{key}.norms"] = np.stack(norms, axis=0)
        else:
            # Ragged, so no position counted from the front lines up. The *suffix* does: every rung
            # ends with the same `<|im_end|> \\n <|im_start|> assistant \\n <think> \\n\\n </think>
            # \\n\\n`, whatever the sentence length. Keeping the last TAIL tokens therefore keeps every
            # readout position anyone is going to ask about, and negative indices mean the same thing
            # here as they do for the aligned ladders.
            payload[f"{key}.values"] = np.stack([v[:, -TAIL:] for v in stack], axis=0)
            payload[f"{key}.norms"] = np.stack([n[-TAIL:] for n in norms], axis=0)
        shape = payload[f"{key}.values"].shape
        log.info(f"{key}: {shape} (rung, stat, token, layer, column)")

        manifest.append(
            {
                "ladder": condition["ladder"],
                "rendering": condition["rendering"],
                "kind": condition["kind"],
                "unit": condition["unit"],
                "aligned": condition["aligned"],
                "varying": condition["varying"],
                "spans": condition["spans"],
                "doses": condition["doses"],
                "tokens": condition["tokens"][0] if condition["aligned"] else condition["tokens"][-1][-1:],
                "tokens_per_rung": [len(row) for row in condition["ids"]],
                "text": condition["texts"][0],
                "shape": list(shape),
            }
        )

    out = Path(args.out)
    np.savez_compressed(
        out,
        manifest=np.array(json.dumps(manifest)),
        meta=np.array(
            json.dumps(
                {
                    "model": args.model,
                    "vectors": str(args.vectors),
                    "layers": list(LAYERS),
                    "stats": ["cosine", "raw"],
                    "concepts": real,
                    "controls": args.controls,
                    "seed": args.seed,
                    "dtype": "float32",
                }
            )
        ),
        **payload,
    )
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--vectors", default="diff.safetensors")
    parser.add_argument("--out", default="dose-readout.npz")
    parser.add_argument("--controls", type=int, default=512, help="random unit directions for the null")
    parser.add_argument("--seed", type=int, default=20260730)
    main(parser.parse_args())
