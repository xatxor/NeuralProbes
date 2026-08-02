#! /usr/bin/env python

"""Distributions of concept readouts, unsteered against steered, over the lmsys steering prompts.

The question: hold a prompt set fixed, steer with one concept vector, and ask how the readout of
*every* concept moves. Two things come out of that, and only the second is a result.

1. THE STEERED CONCEPT SEPARATES. Expected, and at the injection layer it is arithmetic rather than
   evidence: adding `alpha * ||h|| * v` to the residual stream raises `cos(v, h)` by construction, so
   two perfectly separated bells there would demonstrate only that addition works. This is why the
   headline read is taken at a DIFFERENT layer from the injection -- separation at block 25 from an
   injection at block 18 means the perturbation survived seven blocks of processing, which is a fact
   about the model rather than about the arithmetic.

2. HOW MANY OF THE OTHER 1035 MOVE WITH IT. This is selectivity, it has never been measured here, and
   it is the quantitative successor to the side-effect analysis: steering that works is already known
   to move persona and language 1.8-2.8x more than it moves formatting or length, but that came from
   a judge reading text. This measures it in the residual stream directly.

READOUT POSITION. The final prompt token -- the last token of the assistant header, immediately
before generation. Two independent reasons: it is the position the emotion-vector paper measures at
("the ':' token following 'Assistant'"), and it is the position our own sweep selected on the
criterion that a statistic must first be the same function on differently worded prompts.

NO GENERATION. Everything is a prefill forward pass, so there is no sampling noise and no dependence
on decoding parameters; baseline and steered differ only by the hook. That also makes the whole run
deterministic and cheap.

Z-SCORING. Each concept is standardised by its own BASELINE mean and standard deviation across the
prompt set, so the unsteered distribution is N(0,1) by construction and the steered distribution's
displacement is read directly in standard deviations. This is what makes "two bells" the right mental
picture and makes concepts with different natural scales comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("zdist")

# Row index of each block inside the published vector tensor.
ROW = {11: 0, 14: 1, 18: 2, 22: 3, 25: 4}


def build(tokenizer, prompts: list[str]) -> list[torch.Tensor]:
    """Render each prompt as a user turn with the assistant header open.

    :param tokenizer: the model's tokenizer.
    :param prompts: raw prompt strings.

    :return: one token tensor per prompt.
    """
    return [tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                          add_generation_prompt=True, enable_thinking=False,
                                          return_tensors="pt")[0] for text in prompts]


def readout(model, ids: torch.Tensor, layers: list[int], unit: dict[int, torch.Tensor],
            delta: tuple[int, torch.Tensor] | None) -> dict[int, np.ndarray]:
    """One forward pass; return cosine of the final token against every direction, per layer.

    :param model: the loaded model.
    :param ids: token ids for one prompt.
    :param layers: blocks to read.
    :param unit: block -> `[1036, hidden]` unit-normalised directions.
    :param delta: `(block, vector)` to add at that block's output, or None for the baseline.

    :return: block -> `[1036]` cosines.
    """
    handles = []
    if delta is not None:
        block, vector = delta

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vector,) + output[1:]
            return output + vector

        handles.append(model.model.layers[block - 1].register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = model(ids.unsqueeze(0).to(model.device), output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer in layers:
        # hidden_states[k] is the input to block k, so the OUTPUT of block `layer` is index `layer`.
        state = out.hidden_states[layer][0, -1].float()
        # Cosine, not dot product: token norms span two orders of magnitude within one sequence, so a
        # raw projection would rank positions by norm rather than by content.
        state = state / (state.norm() + 1e-6)
        result[layer] = (unit[layer].to(state.dtype) @ state).cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--vectors", type=Path, default=Path("probes-notemplate/diff.safetensors"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--inject-layer", type=int, default=18)
    parser.add_argument("--read-layers", type=int, nargs="+", default=[18, 25])
    parser.add_argument("--pairs", type=int, nargs="+", required=True,
                        help="concept indices to steer with, one arm each")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, -0.5])
    parser.add_argument("--controls", type=int, default=2,
                        help="matched random directions, drawn as the original screen drew them")
    parser.add_argument("--limit", type=int, default=0, help="cap the prompt set for a smoke run")
    parser.add_argument("--out", type=Path, default=Path("analysis/zdist.npz"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    payload = json.loads(args.prompts.read_text())
    prompts, provenance = [], []
    for identifier, entry in sorted(payload.items(), key=lambda kv: int(kv[0])):
        for slot, item in enumerate(entry["prompts"]):
            prompts.append(item["text"])
            provenance.append(f"{identifier}:{slot}")
    if args.limit:
        prompts, provenance = prompts[: args.limit], provenance[: args.limit]
    log.info(f"{len(prompts)} prompts")

    block = load_file(args.vectors)["diff"]
    unit = {layer: torch.nn.functional.normalize(block[ROW[layer]].float(), dim=1).to(args.device)
            for layer in set(args.read_layers) | {args.inject_layer}}
    concepts = unit[args.inject_layer].shape[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device)
    model.eval()
    rendered = build(tokenizer, prompts)

    # Reference norm at the injection block, measured on these prompts rather than assumed, since
    # alpha is defined as a fraction of it.
    norms = []
    with torch.no_grad():
        for ids in rendered[:64]:
            out = model(ids.unsqueeze(0).to(model.device), output_hidden_states=True)
            norms.append(out.hidden_states[args.inject_layer][0, -1].float().norm().item())
    reference = float(np.mean(norms))
    log.info(f"reference norm at block {args.inject_layer}: {reference:.2f}")

    # Controls seeded exactly as the original screen seeded its controls, so these are literally the
    # same random directions that produced the published control floor.
    generator = torch.Generator().manual_seed(0)
    noise = torch.nn.functional.normalize(
        torch.randn(64, unit[args.inject_layer].shape[1], generator=generator), dim=1)

    arms: list[tuple[str, torch.Tensor | None, float]] = [("baseline", None, 0.0)]
    for pair in args.pairs:
        for alpha in args.alphas:
            arms.append((f"pair{pair}@{alpha:+g}", unit[args.inject_layer][pair].cpu(), alpha))
    for index in range(args.controls):
        for alpha in args.alphas:
            arms.append((f"random{index}@{alpha:+g}", noise[index], alpha))
    log.info(f"{len(arms)} arms x {len(prompts)} prompts x {len(args.read_layers)} blocks")

    table = np.zeros((len(arms), len(prompts), len(args.read_layers), concepts), dtype=np.float32)
    for a, (name, vector, alpha) in enumerate(arms):
        delta = None
        if vector is not None:
            delta = (args.inject_layer,
                     (alpha * reference * vector.to(args.device)).to(getattr(torch, args.dtype)))
        for p, ids in enumerate(rendered):
            values = readout(model, ids, args.read_layers, unit, delta)
            for l, layer in enumerate(args.read_layers):
                table[a, p, l] = values[layer]
        log.info(f"[{a + 1}/{len(arms)}] {name} done")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, table=table,
        arms=np.array([name for name, _, _ in arms]),
        alphas=np.array([alpha for _, _, alpha in arms], dtype=np.float32),
        prompts=np.array(provenance), layers=np.array(args.read_layers),
        inject_layer=np.array(args.inject_layer), reference=np.array(reference))
    log.info(f"wrote {args.out}  shape {table.shape}")


if __name__ == "__main__":
    main()
