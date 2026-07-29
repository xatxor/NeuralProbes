#! /usr/bin/env python

"""Project a saved trajectory onto every concept vector, token by token.

Runs the recorded token stream back through the model and reads the residual stream at L18 and L25,
projecting each position onto all 1036 `diff` directions. The projection is a **cosine**, not a dot
product: token norms vary by two orders of magnitude within one sequence, so raw dot products would
mostly measure norm.

Only the transformer body is run, never the LM head. At ten thousand tokens a 151936-wide logits
tensor would cost several gigabytes for something we do not use.
"""

import argparse
import glob
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from model import load

log = logging.getLogger("readout")

VECTORS = "josephofthebread/Qwen3-8B-concept-vectors"
# The file's own manifest records layers [11, 14, 18, 22, 25], so these are the rows we want.
LAYERS = {18: 2, 25: 4}
# Long trajectories go through in pieces carrying a KV cache, which is equivalent to one forward
# pass but never materialises activations for the whole sequence at once.
CHUNK = 2048


def find(name: str) -> str:
    """Locate a file inside the local HF cache.

    :param name: the file's name within the vectors repo.

    :return: an absolute path.
    """
    pattern = f"**/models--{VECTORS.replace('/', '--')}/snapshots/*/{name}"
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / pattern), recursive=True)
    if not matches:
        raise SystemExit(f"{name} is not in the local cache; this box is meant to have it already")
    return matches[0]


def capture(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    slot: int,
    vectors: torch.Tensor,
    into: dict[int, torch.Tensor],
) -> None:
    """Project one block's residual stream onto every direction, as a forward hook.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param slot: index of this layer in the output array.
    :param vectors: unit directions at this layer as `[pair, hidden]` float32.
    :param into: mapping the hook writes this chunk's scores into.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = (output[0] if isinstance(output, tuple) else output).float()
    if not torch.isfinite(state).all():
        raise RuntimeError("non-finite residual stream during readout; fp16 has overflowed")
    norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
    into[slot] = ((state / norm) @ vectors.T)[0]
    # The cosine divides magnitude out, so it is lost unless kept here. The viewer shows it per token
    # and it is the only way to see that norms span two orders of magnitude within one sequence.
    into[-1 - slot] = norm[0, :, 0]


def score(model: Any, ids: list[int], vectors: torch.Tensor) -> np.ndarray:
    """Read every concept off every token of one trajectory.

    :param model: the loaded causal LM.
    :param ids: the recorded token stream.
    :param vectors: unit directions as `[layer, pair, hidden]` float32 on the GPU.

    :return: `[token, layer, pair]` float16 cosines, and `[token, layer]` float32 norms.
    """
    blocks = model.model.layers
    chunk_scores: dict[int, torch.Tensor] = {}
    handles = [
        blocks[layer - 1].register_forward_hook(
            partial(capture, slot=slot, vectors=vectors[slot], into=chunk_scores)
        )
        for slot, layer in enumerate(LAYERS)
    ]
    collected: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    try:
        past, device = None, model.device
        with torch.inference_mode():
            for start in range(0, len(ids), CHUNK):
                piece = ids[start : start + CHUNK]
                output = model.model(
                    input_ids=torch.tensor([piece], device=device),
                    attention_mask=torch.ones(1, start + len(piece), dtype=torch.long, device=device),
                    cache_position=torch.arange(start, start + len(piece), device=device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                collected.append(
                    torch.stack([chunk_scores[slot] for slot in range(len(LAYERS))], dim=1)
                    .half()
                    .cpu()
                    .numpy()
                )
                norms.append(
                    torch.stack([chunk_scores[-1 - slot] for slot in range(len(LAYERS))], dim=1)
                    .float()
                    .cpu()
                    .numpy()
                )
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(collected, axis=0), np.concatenate(norms, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("episodes/stage1"))
    parser.add_argument("--device", default="cuda:0")
    # Must match the dtype the episode was generated in. Replaying a bfloat16 episode in float16
    # reads a residual stream the generation never had.
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--redo", action="store_true", help="rescore episodes that already have scores")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    block = load_file(find("diff.safetensors"))["diff"]
    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    names = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    classes = [row["class_name"] for row in rows]

    vectors = torch.nn.functional.normalize(
        torch.stack([block[index] for index in LAYERS.values()]).float(), dim=-1
    ).to(args.device)
    log.info(f"vectors {tuple(vectors.shape)} at layers {list(LAYERS)}")

    model, _ = load(device=args.device, dtype=args.dtype)

    for offset, record in enumerate(sorted(args.dir.glob("*.json"))):
        if offset % args.shards != args.shard:
            continue
        target = record.with_suffix(".scores.npy")
        # Skipping finished episodes makes the sweep resumable: a shard that dies costs its own
        # remaining work, not the whole run.
        if target.exists() and not args.redo:
            continue
        episode = json.loads(record.read_text())
        ids = episode["ids"]
        scores, magnitude = score(model, ids, vectors)
        np.save(target, scores)
        np.save(record.with_suffix(".norms.npy"), magnitude.astype(np.float32))

        roles = np.array(episode["roles"])
        thinking = scores[roles == "thinking"] if (roles == "thinking").any() else scores
        strength = np.abs(thinking[:, 1, :].astype(np.float32)).mean(axis=0)
        top = np.argsort(-strength)[:5]
        log.info(f"{record.stem}: {scores.shape} -> {target.name}")
        for index in top:
            log.info(f"    L25 {strength[index]:.4f}  {names[index][:58]}  [{classes[index][:28]}]")


if __name__ == "__main__":
    main()
