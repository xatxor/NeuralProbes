#! /usr/bin/env python

import json
import logging
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("jailbreak")


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


def readout(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    position: int,
    vectors: torch.Tensor,
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Project one block's residual stream onto every concept direction, as a forward hook.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param position: index of this layer within `LAYERS`, and the key written to `captured`.
    :param vectors: unit directions at this layer as `[method * pair, hidden]` float32.
    :param captured: mapping the hooks write into, read once per forward by the caller.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = (output[0] if isinstance(output, tuple) else output).float()
    norm = torch.linalg.vector_norm(state, dim=-1)
    captured[position] = ((state @ vectors.T) / norm.unsqueeze(-1).clamp_min(1e-6), norm)


def render(tokenizer: Any, text: str) -> torch.Tensor:
    """Tokenize one user turn through the chat template the vectors were extracted under.

    :param tokenizer: the model's tokenizer.
    :param text: the user message.

    :return: token ids as `[1, tokens]` on the GPU.
    """
    return tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].cuda()


def trace(
    model: Any,
    tokenizer: Any,
    prompt: torch.Tensor,
    vectors: torch.Tensor,
    layers: list[int],
    delta: tuple[int, torch.Tensor] | None,
    args: Namespace,
    seed: int,
) -> dict[str, Any]:
    """Sample one continuation, then read every concept off every token of the whole sequence.

    :param model: the full causal LM, on the GPU in eval mode.
    :param tokenizer: its tokenizer.
    :param prompt: rendered prompt as `[1, tokens]`.
    :param vectors: unit directions as `[layer, method * pair, hidden]` float32 on the GPU.
    :param layers: block numbers to read, one-indexed as the published vectors name them.
    :param delta: block index and already-scaled vector to inject, or None for an unsteered run.
    :param args: parsed arguments; `new_tokens` and `clean` are read.
    :param seed: torch seed set immediately before sampling.

    :return: `scores [tokens, layer, method * pair]` float16, `norms [tokens, layer]` float32,
        `logprobs [tokens]` float32 with the prompt positions set to nan, the token ids, the decoded
        token strings, and the count of prompt tokens.
    """
    blocks = model.model.layers
    handles = [blocks[delta[0]].register_forward_hook(partial(steer, delta=delta[1]))] if delta else []
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            sequence = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=args.new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=20,
                top_p=0.95,
                pad_token_id=model.generation_config.pad_token_id,
            )
        # Detach before the readout so a steered concept does not measure its own injection.
        if delta:
            for handle in handles:
                handle.remove()
            handles = []

        captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        handles += [
            blocks[layer - 1].register_forward_hook(
                partial(readout, position=position, vectors=vectors[position], captured=captured)
            )
            for position, layer in enumerate(layers)
        ]
        with torch.inference_mode():
            logits = model(input_ids=sequence, attention_mask=torch.ones_like(sequence)).logits.float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits; rerun in fp32")
    finally:
        for handle in handles:
            handle.remove()

    width = prompt.shape[1]
    scores = torch.stack([captured[position][0][0] for position in range(len(layers))], dim=1)
    norms = torch.stack([captured[position][1][0] for position in range(len(layers))], dim=1)
    chosen = torch.log_softmax(logits[0, :-1], dim=-1).gather(1, sequence[0, 1:, None]).squeeze(1)
    logprobs = torch.cat([torch.full((1,), float("nan"), device=chosen.device), chosen])
    return {
        "scores": scores.half().cpu().numpy(),
        "norms": norms.cpu().numpy(),
        "logprobs": logprobs.float().cpu().numpy(),
        "tokens": sequence[0].cpu().numpy(),
        "pieces": tokenizer.convert_ids_to_tokens(sequence[0].tolist()),
        "width": width,
        "text": str(tokenizer.decode(sequence[0, width:], skip_special_tokens=True)),
    }


def write(
    result: dict[str, Any], stem: str, pairs: int, layers: list[int], methods: tuple[str, ...], out: Path
) -> None:
    """Write one generation as a token table plus one score file per construction.

    :param result: one `trace` return value.
    :param stem: file name stem identifying behaviour, cell, sample and any steering.
    :param pairs: number of concept pairs, so the method axis can be split.
    :param layers: block numbers, for naming the per-layer norm columns.
    :param methods: construction names, one score file each.
    :param out: directory to write into.

    :return: None.
    """
    total = len(result["tokens"])
    roles = ["prompt"] * result["width"] + ["response"] * (total - result["width"])
    pq.write_table(
        pa.table(
            {
                "position": pa.array(range(total), pa.int32()),
                "token_id": pa.array(result["tokens"], pa.int32()),
                "piece": pa.array(result["pieces"], pa.string()),
                "role": pa.array(roles, pa.string()),
                "logprob": pa.array(result["logprobs"], pa.float32()),
                **{
                    f"norm_L{layer}": pa.array(result["norms"][:, position], pa.float32())
                    for position, layer in enumerate(layers)
                },
            }
        ),
        out / f"{stem}.tokens.parquet",
        compression="zstd",
    )
    for index, method in enumerate(methods):
        block = np.ascontiguousarray(result["scores"][:, :, index * pairs : (index + 1) * pairs])
        pq.write_table(
            pa.table(
                {
                    "position": pa.array(range(total), pa.int32()),
                    "scores": pa.FixedSizeListArray.from_arrays(
                        pa.array(block.reshape(-1), pa.float16()), len(layers) * pairs
                    ),
                }
            ),
            out / f"{stem}_m{index}.parquet",
            compression="zstd",
        )


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
    cells = ("A", "B", "C", "D")

    behaviours = json.loads(args.behaviours.read_text())["behaviours"]
    mine = behaviours[args.shard :: args.shards]
    log.info(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(behaviours)} behaviours")

    stacked = load_file(args.readouts)["readouts"]
    count, depth, pairs, hidden = stacked.shape
    if (count, depth) != (len(methods), len(layers)):
        raise SystemExit(f"{args.readouts} holds {count}x{depth}, expected {len(methods)}x{len(layers)}")
    unit = torch.nn.functional.normalize(stacked.float(), dim=3).permute(1, 0, 2, 3).reshape(depth, -1, hidden).cuda()
    log.info(f"{count} constructions x {depth} layers x {pairs} pairs, {unit.numel() * 4 / 1e6:.0f} MB on device")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    log.info(f"{torch.cuda.get_device_name(0)}, using {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": "cuda"}, attn_implementation="sdpa"
    )
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    delta, tag = None, ""
    if args.steer:
        pair, rest = args.steer.split("@")
        layer, alpha = rest.split("=")
        position = layers.index(int(layer.lstrip("L")))
        reference = 268.51
        direction = unit[position, methods.index("diff") * pairs + int(pair)]
        delta = (int(layer.lstrip("L")) - 1, (float(alpha) * reference * direction).to(model.dtype))
        tag = f"_steer-{pair}@L{layer.lstrip('L')}a{float(alpha):+.2f}"
        log.info(f"steering pair {pair} at L{layer.lstrip('L')} alpha {alpha} against reference {reference}")

    args.out.mkdir(parents=True, exist_ok=True)
    index, done = [], 0
    for behaviour in mine:
        for cell in cells:
            for draw in range(args.samples):
                stem = f"jailbreak_{behaviour['id']}_{cell}_s{draw}{tag}"
                if (args.out / f"{stem}.tokens.parquet").exists():
                    continue
                result = trace(
                    model,
                    tokenizer,
                    render(tokenizer, behaviour["cells"][cell]),
                    unit,
                    layers,
                    delta,
                    args,
                    seed=abs(hash((behaviour["id"], cell, draw))) % (2**31),
                )
                write(result, stem, pairs, layers, methods, args.out)
                index.append(
                    {
                        "stem": stem,
                        "behaviour": behaviour["id"],
                        "half": behaviour["half"],
                        "topic": behaviour["topic"],
                        "tactic": behaviour["tactic"],
                        "cell": cell,
                        "sample": draw,
                        "prompt": behaviour["cells"][cell],
                        "text": result["text"],
                        "width": result["width"],
                        "tokens": len(result["tokens"]),
                        "steer": args.steer or "",
                    }
                )
                done += 1
        rate = done / max(1e-9, time.monotonic() - started)
        log.info(f"{behaviour['id']}: {done} generations, {rate:.2f}/s")

    (args.out / f"index-{args.shard}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "layers": layers,
                "methods": list(methods),
                "pairs": pairs,
                "cells": list(cells),
                "samples": args.samples,
                "new_tokens": args.new_tokens,
                "shard": args.shard,
                "shards": args.shards,
                "steer": args.steer or "",
                "clean": True,
                "generations": done,
                "axes": {"scores": ["token", "layer", "pair"], "norms": ["token", "layer"]},
                "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
            },
            indent=2,
        )
    )
    log.info(f"wrote {args.out}: {done} generations in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="directory to write this shard into")
    parser.add_argument("--behaviours", type=Path, default=Path("behaviours.json"))
    parser.add_argument("--readouts", type=Path, default=Path("readouts.safetensors"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--samples", type=int, default=4, help="rollouts per cell")
    parser.add_argument("--new-tokens", type=int, default=320)
    parser.add_argument("--steer", default="", help="pair@Llayer=alpha, for example 586@L25=0.5")
    main(parser.parse_args())
