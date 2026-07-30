#! /usr/bin/env python

"""Read every concept off every token of the twenty-five chosen demonstrations.

`top25.pdf` shows three responses per example and no indication of *where* in them the concept lives.
The reward-hacking transcripts have that shading; these do not, because `rescreen.py` stored decoded
text and nothing else. This recovers it by pushing the saved text back through the model and reading
the residual stream, which is what `agentic/readout.py` does for agent trajectories.

Two things make the result less straightforward than it looks, and both are recorded per row rather
than silently averaged away.

**The steered arms are partly circular.** They were generated with `alpha * N * v` added at every
token position of one block, so projecting them back onto that same direction at that same block
finds what the hook put there. Three readings are not circular: the baseline arm anywhere, either
steered arm at a block other than the one it was steered at, and the steered arm at its own block
once the constant offset is subtracted. The offset is knowable -- it is `alpha` times that layer's
reference norm along the unit direction -- so `injected_at` is written into the output and the
subtraction is left to the analysis rather than baked in here.

**Projection is onto the unit residual, not the raw one.** Token norms vary by two orders of
magnitude inside one sequence, so raw dot products would rank tokens by norm rather than by content.

Only the transformer body runs. The LM head is 151936 wide and would dominate both time and memory
for an output nobody reads.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("highlight")

# Layer number to its index along the vector tensor's first axis, which is ordered as the vectors
# were written: blocks 11, 14, 18, 22, 25 of 36.
LAYERS = {11: 0, 14: 1, 18: 2, 22: 3, 25: 4}
ARMS = ("response_baseline", "response_toward_concept", "response_toward_antagonist")


def capture(store: dict[int, torch.Tensor], slot: int, directions: torch.Tensor) -> Any:
    """Build a forward hook that projects one block's output onto every direction.

    :param store: mapping the hook writes its result into.
    :param slot: index of this block along the vector tensor.
    :param directions: unit concept directions for this block, `[pairs, hidden]`.

    :return: a hook function; returning None leaves the block's output untouched.
    """

    def hook(module: Any, inputs: Any, output: Any) -> None:
        state = output[0] if isinstance(output, tuple) else output
        state = state[0].float()
        store[slot] = (state / state.norm(dim=-1, keepdim=True)) @ directions.T

    return hook


def score(model: Any, ids: list[int], vectors: torch.Tensor, device: str) -> np.ndarray:
    """Read every concept off every token of one sequence.

    :param model: the loaded body, in eval mode.
    :param ids: the token stream, prompt and response together.
    :param vectors: unit directions, `[layers, pairs, hidden]`.
    :param device: where the model lives.

    :return: `[token, layer, pair]` float16.
    """
    store: dict[int, torch.Tensor] = {}
    blocks = model.model.layers
    handles = [
        blocks[layer - 1].register_forward_hook(capture(store, slot, vectors[slot]))
        for layer, slot in LAYERS.items()
        if layer - 1 < len(blocks)
    ]
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    ordered = [store[slot] for slot in sorted(store)]
    return torch.stack(ordered, dim=1).to(torch.float16).cpu().numpy()


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    selected = json.loads(args.source.read_text())
    log.info(f"{len(selected)} examples x {len(ARMS)} arms = {len(selected) * len(ARMS)} sequences")

    block = load_file(args.vectors)["diff"]
    # Normalised once here rather than per sequence: the projection is a cosine, and the vectors do
    # not change between sequences.
    vectors = torch.nn.functional.normalize(torch.from_numpy(block).float(), dim=-1).to(args.device)
    log.info(f"vectors {tuple(vectors.shape)} from {args.vectors}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bf16, matching the dtype the vectors were extracted under and the dtype the responses were
    # generated under. `agentic/model.py` pins fp16 and refuses to run where bf16 exists, which was
    # the right call for V100s and the wrong one here; that decision is re-made rather than bypassed.
    assert torch.cuda.is_bf16_supported(), "expected a bf16-capable card; re-check the dtype choice"
    highest = max(LAYERS)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa"
    )
    # Nothing above the deepest probed block contributes, and the LM head never runs.
    model.model.layers = model.model.layers[:highest]
    model.eval()
    log.info(f"model truncated to {len(model.model.layers)} blocks, {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    tensors: dict[str, np.ndarray] = {}
    index = []
    for entry in selected:
        prompt_ids = tokenizer.encode(entry["prompt_templated"], add_special_tokens=False)
        for arm in ARMS:
            reply_ids = tokenizer.encode(entry[arm], add_special_tokens=False)
            ids = prompt_ids + reply_ids
            if len(ids) > args.max_tokens:
                log.warning(f"rank {entry['rank']} {arm}: {len(ids)} tokens, truncating to {args.max_tokens}")
                ids = ids[: args.max_tokens]
            scores = score(model, ids, vectors, args.device)
            key = f"{entry['rank']:02d}:{arm}"
            tensors[key] = scores
            index.append(
                {
                    "key": key,
                    "rank": entry["rank"],
                    "arm": arm,
                    "pair": entry["pair"],
                    "class": entry["class"],
                    "concept": entry["concept"],
                    "antagonist": entry["antagonist"],
                    # Which block carried the intervention, so the analysis knows which readings are
                    # circular. The baseline arm had none.
                    "injected_at": None if arm == "response_baseline" else entry["layer"],
                    "alpha": 0.0 if arm == "response_baseline" else (0.5 if "concept" in arm else -0.5),
                    "prompt_tokens": len(prompt_ids),
                    "reply_tokens": len(ids) - len(prompt_ids),
                    "shape": list(scores.shape),
                }
            )
            log.info(f"rank {entry['rank']:>2} {arm:<26} {scores.shape} tokens={len(ids)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.out),
        metadata={
            "manifest": json.dumps(
                {
                    "model": args.model,
                    "vectors": str(args.vectors),
                    "layers": sorted(LAYERS),
                    "axes": ["token", "layer", "pair"],
                    "dtype": "float16",
                    "normalisation": "residual divided by its own L2 norm per token, then cosine",
                    "n_sequences": len(tensors),
                    "index": index,
                }
            )
        },
    )
    total = sum(value.nbytes for value in tensors.values())
    log.info(f"wrote {args.out}: {len(tensors)} sequences, {total/1e6:.1f} MB")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("top25.json"))
    parser.add_argument("--vectors", type=Path, default=Path("diff.safetensors"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--out", type=Path, default=Path("top25-readout.safetensors"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=4096, help="guard against a runaway sequence")
    main(parser.parse_args())
