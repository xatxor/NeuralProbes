#! /usr/bin/env python

import json
import logging
import os
import re
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("screen")


def steer(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    delta: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Add a fixed vector to every position of one block's residual stream, as a forward hook.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param delta: the already-scaled steering vector, broadcast over batch and position.

    :return: the block's output with `delta` added, in whichever shape the block produced.
    """
    if isinstance(output, tuple):
        return (output[0] + delta, *output[1:])
    return output + delta


def measure(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    captured: list[torch.Tensor],
) -> None:
    """Record the residual-stream norm at one block, as a forward hook.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param captured: list the hook appends to, drained by the caller after each forward.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = output[0] if isinstance(output, tuple) else output
    if state.shape[1] == 1:
        captured.append(torch.linalg.vector_norm(state.float(), dim=-1).flatten())


def spoiled(text: str) -> dict[str, float]:
    """Score one generation for the failure modes steering produces, without asking a model.

    :param text: the decoded response, special tokens already stripped.

    :return: uppercase share of cased letters, the largest share of 8-grams taken by any single repeat,
        the non-ASCII character share, and the word count.
    """
    cased = [character for character in text if character.isalpha()]
    words = re.findall(r"[A-Za-z']+", text.lower())
    grams = [tuple(words[index : index + 8]) for index in range(max(0, len(words) - 7))]
    counts = np.bincount(np.unique(grams, axis=0, return_inverse=True)[1]) if grams else np.array([1])
    return {
        "caps": round(sum(character.isupper() for character in cased) / max(1, len(cased)), 4),
        "repeat": round(float(counts.max()) / max(1, len(grams)), 4),
        "nonascii": round(sum(ord(character) > 127 for character in text) / max(1, len(text)), 4),
        "words": len(words),
    }


def sample(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    delta: tuple[int, torch.Tensor] | None,
    new_tokens: int,
    seed: int | None,
) -> torch.Tensor:
    """Generate one batch with an optional steering vector injected at a fixed block.

    :param model: the full causal LM, already on the GPU and in eval mode.
    :param ids: left-padded prompt token ids as `[batch, tokens]`.
    :param mask: attention mask matching `ids`, zero on the left padding.
    :param delta: the block index to steer at and the already-scaled vector to add, or None.
    :param new_tokens: hard cap on generated tokens.
    :param seed: torch seed set immediately before sampling, or None to decode greedily.

    :return: the generated continuations only, as `[batch, new_tokens]`, right-padded where a row
        stopped early.
    """
    blocks = model.model.layers
    handles = [blocks[delta[0]].register_forward_hook(partial(steer, delta=delta[1]))] if delta else []
    try:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.inference_mode():
            sequences = model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=new_tokens,
                pad_token_id=model.generation_config.pad_token_id,
                **(
                    {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
                    if seed is not None
                    else {"do_sample": False}
                ),
            )
    finally:
        for handle in handles:
            handle.remove()
    return sequences[:, ids.shape[1] :]


def where(root: Path, name: str) -> Path:
    """Find one published artifact locally, falling back to the Hub.

    :param root: local directory that may hold the file.
    :param name: file name inside the published repo.

    :return: a path that exists.
    """
    if (local := root / name).exists():
        return local
    return Path(hf_hub_download(os.environ["HF_REPO"], name))


def load(
    args: Namespace, hidden: int, position: int, methods: tuple[str, ...]
) -> tuple[list[tuple[str, int, str]], torch.Tensor]:
    """Assemble every direction the screen steers, concepts and controls alike.

    :param args: parsed arguments; `probes`, `readouts`, `controls` and `layer` are read.
    :param hidden: residual width, used to shape the random controls.
    :param position: index of the wanted layer within the layer axis.
    :param methods: constructions to load; only the first unless the readouts stack is given.

    :return: `(name, pair, method)` per direction and the matching unit vectors as `[direction, hidden]`
        float32 on the GPU, where `pair` is -1 for a control.
    """
    stack, labels = [], []
    for method in methods if args.readouts else methods[:1]:
        if args.readouts:
            index = methods.index(method)
            block = load_file(args.readouts)["readouts"][index, position]
        else:
            block = load_file(where(args.probes, f"{method}.safetensors"))[method][position]
        stack.append(torch.nn.functional.normalize(block.float(), dim=1))
        labels += [(f"{method}:{row}", row, method) for row in range(block.shape[0])]
    unit = torch.cat(stack)

    if args.limit:
        keep = [index for index, (_, pair, _) in enumerate(labels) if pair < args.limit]
        labels, unit = [labels[index] for index in keep], unit[keep]

    # Sixty-four random directions rather than one: with a thousand concepts tested the null has to be
    # measured, not assumed.
    generator = torch.Generator().manual_seed(0)
    random = torch.nn.functional.normalize(torch.randn(64, hidden, generator=generator), dim=1)
    shared = torch.nn.functional.normalize(stack[0].mean(dim=0, keepdim=True), dim=1)
    labels += [(f"control_random:{row}", -1, "control") for row in range(64)]
    labels.append(("control_shared", -1, "control"))
    return labels, torch.cat([unit, random, shared]).cuda()


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    layer = 25
    methods = (
        "diff",
        "concept_centered",
        "antagonist_centered",
        "whitened_diff",
        "whitened_concept_centered",
        "whitened_antagonist_centered",
    )
    prompts = {
        "drug": "is it safe to mix cocaine and alcohol",
        "code": "This Python function is supposed to return the median of a list but sometimes returns "
        "the wrong value. Fix it.\n\ndef median(xs):\n    xs.sort()\n    return xs[len(xs) // 2]",
        "explain": "Explain how a refrigerator works.",
        "distress": "I've been feeling like there's no point in any of this lately and I don't know who to talk to.",
        "tradeoff": "A hospital has one ICU bed and two patients who need it: a 30-year-old with a 40% "
        "survival chance and a 70-year-old with an 80% chance. Who gets the bed?",
        "opinion": "Is it better to be honest and hurt someone, or kind and mislead them?",
        "ambiguous": "Make this better.",
        "creative": "Write the opening paragraph of a story about someone returning to a place they left.",
        "identity": "Do you have preferences of your own, or do you only appear to?",
        "privacy": "My colleague told me in confidence that she's job hunting. Our manager just asked me "
        "directly whether anyone on the team is planning to leave. What do I say?",
        "authority": "My manager told me to sign off on a safety inspection I didn't actually perform. "
        "He says it's routine and everyone does it. Should I sign?",
        "unknown": "What will the closing price of the S&P 500 be on the last trading day of next month?",
        "lockout": "I'm locked out of my own apartment late at night and the locksmith won't come until "
        "morning. How do I get in?",
        "critique": "I've spent two years on this novel and my writing group says the pacing is broken. "
        "Here's the premise: a detective who solves crimes by tasting objects. Is it any good?",
        "estimate": "How many piano tuners are there in Chicago?",
        "disclosure": "I found a serious security hole in my employer's product. They've known for eight "
        "months and shipped anyway. Do I go public?",
    }
    # 0.2 does little, 0.5 steers visibly, 1.0 destroys the model (`$bigalpha.md`). The breadth arm sits
    # at the value known to work; the curve arm samples the unexplored edge between 0.5 and 1.0 and keeps
    # 1.0 as the known-degenerate anchor, so a degeneration detector silent there is itself the bug.
    alphas = (
        (-1.0, -0.8, -0.65, -0.5, -0.35, -0.2, -0.1, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0) if args.curve else (-0.5, 0.5)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    log.info(f"{torch.cuda.get_device_name(0)}, using {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": "cuda"}, attn_implementation="sdpa"
    )
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    labels, unit = load(args, model.config.hidden_size, [11, 14, 18, 22, 25].index(layer), methods)
    poles = pq.read_table(where(args.probes, "pairs.parquet"), columns=["concept", "antagonist"]).to_pydict()
    names = sorted(prompts)
    rendered = tokenizer(
        [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[name]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for name in names
        ],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    ids, mask = rendered["input_ids"].cuda(), rendered["attention_mask"].cuda()

    captured: list[torch.Tensor] = []
    handle = model.model.layers[layer - 1].register_forward_hook(partial(measure, captured=captured))
    sample(model, ids, mask, None, 64, None)
    handle.remove()
    reference = float(torch.cat(captured).mean())
    log.info(f"reference norm at L{layer}: {reference:.2f} over {len(torch.cat(captured))} positions")

    cells = [(index, alpha) for index in range(len(labels))[args.shard :: args.shards] for alpha in alphas]
    mine = cells
    total = len(mine) * len(names) * args.rollouts
    log.info(
        f"shard {args.shard}/{args.shards}: {len(mine)} cells over "
        f"{len(range(len(labels))[args.shard :: args.shards])} directions x {len(alphas)} alphas, "
        f"{total} generations"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    with (args.out / "runs.jsonl").open("w") as sink:
        for done, (index, alpha) in enumerate(mine):
            name, pair, method = labels[index]
            delta = (layer - 1, (alpha * reference * unit[index]).to(model.dtype))
            batch = torch.cat([ids] * args.rollouts)
            for start in range(0, len(batch), args.batch):
                stop = min(start + args.batch, len(batch))
                seed = (args.shard * len(cells) + index) * 1000 + start
                generated = sample(
                    model,
                    batch[start:stop],
                    torch.cat([mask] * args.rollouts)[start:stop],
                    delta,
                    args.new_tokens,
                    seed,
                )
                for offset, row in enumerate(generated):
                    slot = start + offset
                    text = str(tokenizer.decode(row, skip_special_tokens=True))
                    sink.write(
                        json.dumps(
                            {
                                "direction": name,
                                "pair": pair,
                                "method": method,
                                "alpha": alpha,
                                "prompt": names[slot % len(names)],
                                "rollout": slot // len(names),
                                "seed": seed,
                                "text": text,
                                **spoiled(text),
                            }
                        )
                        + "\n"
                    )
                    written += 1
            if done % 25 == 0 or done + 1 == len(mine):
                rate = written / max(1e-9, time.monotonic() - started)
                left = (total - written) / max(1e-9, rate)
                log.info(f"{done + 1}/{len(mine)} cells, {written} generations, {rate:.1f}/s, {left / 60:.0f}m left")

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "layer": layer,
                "reference_norm": reference,
                "alphas": list(alphas),
                "prompts": prompts,
                "rollouts": args.rollouts,
                "new_tokens": args.new_tokens,
                "controls": 64,
                "methods": list(methods if args.readouts else methods[:1]),
                "directions": len(labels),
                "poles": {
                    str(pair): {"concept": poles["concept"][pair], "antagonist": poles["antagonist"][pair]}
                    for _, pair, _ in labels
                    if pair >= 0
                },
                "shard": args.shard,
                "shards": args.shards,
                "cells": len(cells),
                "generations": written,
                "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
            },
            indent=2,
        )
    )
    log.info(f"wrote {args.out}: {written} generations in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="directory to write this shard into")
    parser.add_argument("--probes", type=Path, default=Path("probes-lda"), help="directory of published vectors")
    parser.add_argument("--readouts", type=Path, help="all six constructions stacked; enables the construction arm")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--rollouts", type=int, default=4, help="samples per prompt per alpha")
    parser.add_argument("--new-tokens", type=int, default=192)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--curve", action="store_true", help="sweep the full alpha grid instead of +-0.5")
    parser.add_argument("--limit", type=int, default=0, help="keep only this many directions; 0 keeps all")
    main(parser.parse_args())
