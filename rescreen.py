#! /usr/bin/env python

"""Generate the re-screen: every concept vector steered on prompts drawn from its own class.

The original screen tested all 1036 vectors on the same sixteen generic prompts, so a vector could
look inert merely because no prompt gave it room to show. Here each vector meets the eight prompts
curated for its own class.

Three arms run per vector: the residual stream is pushed along the direction, pushed against it,
and had the direction projected out of it. The first two form a blind pair for the judge; ablation
has no mirror, so it is judged against an unsteered baseline that is generated once and reused,
being independent of which layer we intervene at.

Every arm of a given (direction, prompt, seed) shares one seed, so a pair differs by the
intervention and not by sampling noise. Without that the comparison measures nothing.
"""

import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("rescreen")

LAYERS = [11, 14, 18, 22, 25]


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


def ablate(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    direction: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Project one direction out of the residual stream, as a forward hook.

    This is `h - (h . v) v`, removing whatever component the stream carries along the concept,
    rather than adding a fixed amount of it. The projection is computed in float32: in fp16 the dot
    product over 4096 dimensions loses enough precision that the residual is visibly non-orthogonal.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param direction: a unit vector in the residual basis.

    :return: the block's output with the direction removed, in whichever shape the block produced.
    """
    state = output[0] if isinstance(output, tuple) else output
    flat = state.float()
    flat = flat - (flat @ direction).unsqueeze(-1) * direction
    stripped = flat.to(state.dtype)
    if isinstance(output, tuple):
        return (stripped, *output[1:])
    return stripped


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


def sample(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    hook: tuple[int, Any] | None,
    new_tokens: int,
    seed: int | None,
) -> torch.Tensor:
    """Generate one batch with an optional intervention installed at a fixed block.

    :param model: the full causal LM, already on the GPU and in eval mode.
    :param ids: left-padded prompt token ids as `[batch, tokens]`.
    :param mask: attention mask matching `ids`, zero on the left padding.
    :param hook: the block index and the already-bound forward hook to install, or None.
    :param new_tokens: hard cap on generated tokens.
    :param seed: torch seed set immediately before sampling, or None to decode greedily.

    :return: the generated continuations only, as `[batch, new_tokens]`.
    """
    handles = [model.model.layers[hook[0]].register_forward_hook(hook[1])] if hook else []
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

    A container starts empty, so the vectors are either shipped in as an input or fetched. Fetching
    keeps the job spec small; the local branch is what makes the script usable on the box.

    :param root: local directory that may hold the file.
    :param name: file name inside the published repo.

    :return: a path that exists.
    """
    if (local := root / name).exists():
        return local
    return Path(hf_hub_download(os.environ["HF_REPO"], name))


def directions(probes: Path, position: int, randoms: int) -> tuple[list[dict], torch.Tensor]:
    """Assemble every direction the re-screen drives, concepts and controls alike.

    Only `diff` is used. It is the sole construction ever validated behaviourally -- both the
    original screen and the steering work ran on it -- so the other five would be untested
    interventions dressed as replications.

    :param probes: directory holding the published vectors.
    :param position: index of the wanted layer within the layer axis.
    :param randoms: how many random control directions to include.

    :return: one descriptor per direction and the matching unit vectors as `[direction, hidden]`.
    """
    block = load_file(where(probes, "diff.safetensors"))["diff"][position]
    unit = torch.nn.functional.normalize(block.float(), dim=1)
    described = [{"name": f"diff:{row}", "pair": row, "kind": "concept"} for row in range(unit.shape[0])]

    # Seeded exactly as the original screen seeded its controls, so the first four random directions
    # here are the first four random directions there.
    generator = torch.Generator().manual_seed(0)
    noise = torch.nn.functional.normalize(torch.randn(64, unit.shape[1], generator=generator), dim=1)[:randoms]
    shared = torch.nn.functional.normalize(unit.mean(dim=0, keepdim=True), dim=1)

    described += [{"name": f"control_random:{row}", "pair": -1, "kind": "control"} for row in range(randoms)]
    described.append({"name": "control_shared", "pair": -1, "kind": "control"})
    return described, torch.cat([unit, noise, shared]).cuda()


def classes(pairs: Path, catalogue: Path) -> list[int]:
    """Map each concept pair to the id of the class it belongs to.

    :param pairs: `pairs.parquet`, giving every pair its class name.
    :param catalogue: `classes.json`, fixing the order class ids refer to.

    :return: class id per pair, indexed by pair.
    """
    names = json.loads(catalogue.read_text())["classes"]
    order = {name: index for index, name in enumerate(names)}
    return [order[name] for name in pq.read_table(pairs, columns=["class_name"]).column("class_name").to_pylist()]


def plan(
    described: list[dict], of_class: list[int], prompts: dict[int, list[dict]], args: Namespace
) -> list[tuple[int, list[tuple[int, int]], int]]:
    """Lay out the work as chunks small enough to batch, balanced across shards.

    A control faces all 1184 prompts while a concept faces eight, so sharding by direction would
    leave one worker with two hundred times the work of another. Chunks are the unit instead.

    :param described: direction descriptors.
    :param of_class: class id per pair.
    :param prompts: class id to its curated prompts.
    :param args: parsed arguments; `chunk`, `seeds` and `control_seeds` are read.

    :return: `(direction index, [(class id, prompt index, draw)], seeds)` per chunk. The draw index
        travels with each slot rather than being recovered from its offset, because a chunk boundary
        need not fall on a multiple of the seed count.
    """
    everything = [(identifier, index) for identifier in sorted(prompts) for index in range(len(prompts[identifier]))]
    chunks = []
    for index, entry in enumerate(described):
        if entry["kind"] == "control":
            slots, seeds = everything, args.control_seeds
        else:
            identifier = of_class[entry["pair"]]
            slots = [(identifier, slot) for slot in range(len(prompts.get(identifier, [])))]
            seeds = args.seeds
        if not slots:
            continue
        expanded = [(identifier, slot, draw) for identifier, slot in slots for draw in range(seeds)]
        for start in range(0, len(expanded), args.chunk):
            chunks.append((index, expanded[start : start + args.chunk], seeds))
    return chunks


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    position = LAYERS.index(args.layer)

    curated = json.loads(args.prompts.read_text())
    prompts = {int(key): value["prompts"] for key, value in curated.items() if value["prompts"]}
    total_prompts = sum(len(value) for value in prompts.values())
    log.info(f"{len(prompts)} classes carry prompts, {total_prompts} prompts in all")

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

    described, unit = directions(args.probes, position, args.randoms)
    of_class = classes(where(args.probes, "pairs.parquet") if not args.pairs.exists() else args.pairs,
                        args.classes)
    log.info(f"{len(described)} directions at L{args.layer}")

    rendered = {
        (identifier, slot): tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for identifier, rows in prompts.items()
        for slot, row in enumerate(rows)
    }

    captured: list[torch.Tensor] = []
    handle = model.model.layers[args.layer - 1].register_forward_hook(partial(measure, captured=captured))
    warm = tokenizer(list(rendered.values())[:16], add_special_tokens=False, padding=True, return_tensors="pt")
    sample(model, warm["input_ids"].cuda(), warm["attention_mask"].cuda(), None, 64, None)
    handle.remove()
    reference = float(torch.cat(captured).mean())
    log.info(f"reference norm at L{args.layer}: {reference:.2f} over {len(torch.cat(captured))} positions")

    chunks = plan(described, of_class, prompts, args)
    mine = chunks[args.shard :: args.shards]
    log.info(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(chunks)} chunks")

    args.out.mkdir(parents=True, exist_ok=True)
    sink = (args.out / "runs.jsonl").open("w")
    written = 0

    if args.baseline:
        # The baseline is one corpus, not one per shard: sharding it here stops ten workers each
        # generating the whole thing and writing ten copies of every row.
        starts = list(range(0, len(rendered), args.chunk))[args.shard :: args.shards]
        log.info(f"baseline: {len(starts)} of {len(range(0, len(rendered), args.chunk))} chunks on this shard")
        for start in starts:
            slots = list(rendered)[start : start + args.chunk]
            for draw in range(args.seeds):
                batch = tokenizer([rendered[slot] for slot in slots], add_special_tokens=False,
                                  padding=True, return_tensors="pt")
                seed = abs(hash(("baseline", start, draw))) % (2**31)
                out = sample(model, batch["input_ids"].cuda(), batch["attention_mask"].cuda(),
                             None, args.new_tokens, seed)
                for offset, row in enumerate(out):
                    identifier, slot = slots[offset]
                    sink.write(json.dumps({
                        "direction": "baseline", "pair": -1, "kind": "baseline", "arm": "baseline",
                        "layer": None, "class_id": identifier, "prompt": slot, "seed": draw,
                        "text": str(tokenizer.decode(row, skip_special_tokens=True)),
                    }) + "\n")
                    written += 1
        log.info(f"baseline done: {written} generations")

    for done, (index, expanded, seeds) in enumerate(mine):
        entry = described[index]
        vector = unit[index]
        delta = (args.alpha * reference * vector).to(model.dtype)
        batch = tokenizer([rendered[(identifier, slot)] for identifier, slot, _ in expanded],
                          add_special_tokens=False, padding=True, return_tensors="pt")
        ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
        # One seed for the whole chunk, reused across all three arms: the batch composition is
        # identical each time, so a plus/minus pair differs by the intervention alone.
        seed = abs(hash((entry["name"], index, done))) % (2**31)

        arms = {
            "plus": (args.layer - 1, partial(steer, delta=delta)),
            "minus": (args.layer - 1, partial(steer, delta=-delta)),
            "ablate": (args.layer - 1, partial(ablate, direction=vector)),
        }
        for arm, hook in arms.items():
            out = sample(model, ids, mask, hook, args.new_tokens, seed)
            for offset, row in enumerate(out):
                identifier, slot, draw = expanded[offset]
                sink.write(json.dumps({
                    "direction": entry["name"], "pair": entry["pair"], "kind": entry["kind"], "arm": arm,
                    "layer": args.layer, "class_id": identifier, "prompt": slot,
                    "seed": draw, "batch_seed": seed,
                    "text": str(tokenizer.decode(row, skip_special_tokens=True)),
                }) + "\n")
                written += 1

        if done % 50 == 0 or done + 1 == len(mine):
            rate = written / max(1e-9, time.monotonic() - started)
            left = (len(mine) - done - 1) * len(expanded) * len(arms) / max(1e-9, rate)
            log.info(f"{done + 1}/{len(mine)} chunks, {written} generations, {rate:.1f}/s, {left / 60:.0f}m left")

    sink.close()
    (args.out / "manifest.json").write_text(json.dumps({
        "model": args.model, "layer": args.layer, "reference_norm": reference, "alpha": args.alpha,
        "arms": ["plus", "minus", "ablate"], "seeds": args.seeds, "control_seeds": args.control_seeds,
        "randoms": args.randoms, "directions": len(described), "prompts": total_prompts,
        "shard": args.shard, "shards": args.shards, "generations": written,
        "new_tokens": args.new_tokens, "construction": "diff", "enable_thinking": False,
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
    }, indent=2))
    log.info(f"wrote {args.out}: {written} generations in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--pairs", type=Path, default=Path("probes-lda/pairs.parquet"))
    parser.add_argument("--probes", type=Path, default=Path("probes-lda"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layer", type=int, default=18, choices=LAYERS)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=4, help="samples per prompt for concepts")
    parser.add_argument("--control-seeds", type=int, default=2, help="samples per prompt for controls")
    parser.add_argument("--randoms", type=int, default=4, help="random control directions")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=32, help="sequences per generate call")
    parser.add_argument("--new-tokens", type=int, default=192)
    parser.add_argument("--baseline", action="store_true", help="also emit the unsteered baseline")
    main(parser.parse_args())
