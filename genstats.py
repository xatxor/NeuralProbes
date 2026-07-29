#! /usr/bin/env python

import json
import logging
import os
import re
from argparse import ArgumentParser, Namespace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

log = logging.getLogger("genvectors")


def explode(
    batch: dict[str, list[Any]],
    tokenizer: Any,
    pairs: dict[tuple[str, str], int],
) -> dict[str, list[Any]]:
    """Split each corpus row into its two opposing stories and tokenize them.

    :param batch: columns of one batch of corpus rows, as `datasets` passes them to a batched `map`.
    :param tokenizer: tokenizer of the model being probed; stories are encoded without special
        tokens, since the chat scaffolding is added later in `collate`.
    :param pairs: index of every `(concept, antagonist)` label pair, giving each its row number in
        the ontology.

    :return: columns of the exploded batch: `input_ids`, `group`, `label_hit`, `length`, and
        `banned`, the token positions spelling out either pole label. Rows whose text is empty are
        dropped, so the returned columns are shorter than twice the input.
    """
    texts, groups, hits, labels = [], [], [], []
    for index, number in enumerate(batch["pair_number"]):
        key = (batch["concept"][index], batch["antagonist"][index])
        if key not in pairs:
            continue
        for side, column in enumerate(("concept_text", "antagonist_text")):
            text = batch[column][index]
            if not text or not text.strip():
                continue
            texts.append(text)
            groups.append((pairs[key] * 2 + side) * 2 + number % 2)
            hits.append(int(key[side].lower() in text.lower()))
            labels.append(key)

    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=1024,
        return_offsets_mapping=True,
    )
    banned = []
    for text, pair, offsets in zip(texts, labels, encoded["offset_mapping"]):
        forms = {form for label in pair for form in re.split(r"[\s/_-]+", label) if len(form) >= 4}
        forms.update(label for label in pair if len(label) >= 3)
        spans = [m.span() for f in forms for m in re.finditer(re.escape(f), text, re.IGNORECASE)]
        banned.append(
            [
                position
                for position, (start, end) in enumerate(offsets)
                if end > start and any(start < stop and end > begin for begin, stop in spans)
            ]
        )
    return {
        "input_ids": encoded["input_ids"],
        "group": groups,
        "label_hit": hits,
        "length": [len(ids) for ids in encoded["input_ids"]],
        "banned": banned,
    }


def collate(
    rows: list[dict[str, Any]], prefix: list[int], suffix: list[int], pad: int, skip: int
) -> dict[str, torch.Tensor]:
    """Assemble one padded batch and the weight matrix that pools it.

    :param rows: the stories in this batch, as `explode` produced them.
    :param prefix: token ids preceding every story, from the chat template.
    :param suffix: token ids following every story.
    :param pad: token id used to pad short rows; masked out, so its value is irrelevant.
    :param skip: how many leading story tokens to leave out of the pooling mean, as in the paper.
        Clamped per story so at least 16 positions always survive.

    :return: `ids`, `mask` and `weights` for the forward pass, plus the per-story `group`, `length`,
        `label_hit`, and `short` (whether that story's skip window had to be clamped).
    """
    lengths = [len(row["input_ids"]) for row in rows]
    start = len(prefix)
    width = start + max(lengths) + len(suffix)

    ids = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    weights = np.zeros((len(rows), 1, width), dtype=np.float32)
    clamped = [False] * len(rows)

    for position, (row, length) in enumerate(zip(rows, lengths)):
        ids[position, :start] = prefix
        ids[position, start : start + length] = row["input_ids"]
        ids[position, start + length : start + length + len(suffix)] = suffix
        mask[position, : start + length + len(suffix)] = 1

        offset = min(skip, max(0, length - 16))
        clamped[position] = offset < skip
        selector = np.zeros(length, dtype=np.float32)
        selector[offset:] = 1.0
        if banned := row["banned"]:
            kept = selector.copy()
            kept[[index for index in banned if index < length]] = 0.0
            if kept.sum() > 0:
                selector = kept
        weights[position, 0, start : start + length] = selector / selector.sum()

    return {
        "ids": torch.from_numpy(ids),
        "mask": torch.from_numpy(mask),
        "weights": torch.from_numpy(weights),
        "group": torch.tensor([row["group"] for row in rows], dtype=torch.int64),
        "length": torch.tensor(lengths, dtype=torch.int64),
        "label_hit": torch.tensor([row["label_hit"] for row in rows], dtype=torch.int64),
        "short": torch.tensor(clamped, dtype=torch.bool),
    }


def capture(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    position: int,
    pooling: list[torch.Tensor],
    captured: dict[int, torch.Tensor],
) -> None:
    """Pool one block's residual stream, in place, as a forward hook.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: what the block returned. Transformer blocks return either the hidden states or a
        tuple whose first element is the hidden states, so both are accepted.
    :param position: index of this layer among those being read, and the key written to `captured`.
    :param pooling: single-element list holding the current batch's `[batch, 1, tokens]` pooling
        weights. A list rather than a tensor so the caller can swap in new weights each batch
        without re-registering hooks.
    :param captured: mapping the hooks write into, read once per batch by the caller.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = output[0] if isinstance(output, tuple) else output
    captured[position] = torch.bmm(pooling[0].to(state.dtype), state).float().squeeze(1)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    ontology = json.loads(
        Path(hf_hub_download("AntonKorznikov/feature_stories", "ontology.json", repo_type="dataset")).read_text()
    )
    pairs = sorted(
        ({"class_name": entry["name"], **pair} for entry in ontology["classes"] for pair in entry["pairs"]),
        key=lambda pair: (pair["concept"], pair["antagonist"]),
    )
    log.info(f"ontology: {len(pairs)} concept pairs across {len(ontology['classes'])} classes")

    config = AutoConfig.from_pretrained(args.model)
    hidden, depth = config.hidden_size, config.num_hidden_layers
    layers = sorted({round(value * depth) if value < 1.0 else int(value) for value in map(float, args.layers)})
    log.info(f"layers: blocks {layers} of {depth} ({', '.join(args.layers)})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # No chat scaffolding. Every text used to be prefixed with "Write a short story.", which is
    # wrong for most of this corpus: it is ten genres in equal tenths -- memo, news, diary, speech,
    # case study, letter, dialogue, monologue, fable, third-person narrative -- across five languages
    # in equal fifths, so the instruction misdescribed the genre for roughly 80% of rows and was in
    # the wrong language for 80% of them. Both poles carried the same prefix, so most of its effect
    # cancelled in the difference, but it put every activation in a state the model is never actually
    # in, and that is not something to leave in place while asking why the directions fail to
    # transfer.
    #
    # The suffix was `eos_token` and is dropped with it. It never mattered either way: attention is
    # causal and every pooled position is a text token, all of which precede it.
    #
    # Dropping the prefix leaves the first text token without any context, which is harmless here
    # because `--skip-tokens` already discards the leading 50 positions of every text.
    prefix: list[int] = []
    suffix: list[int] = []
    log.info(f"context: raw text, no chat template, {args.skip_tokens} leading tokens skipped")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data = load_dataset("AntonKorznikov/feature_stories", split="train")
    data = data.shard(num_shards=args.shards, index=args.shard, contiguous=False)
    # Nothing is filtered: every language and every genre goes in, as before. The message used to
    # say "after filtering", which described a step that does not exist.
    log.info(f"shard {args.shard}/{args.shards}: {len(data)} of {len(data) * args.shards} rows, unfiltered")

    data = data.map(
        explode,
        batched=True,
        remove_columns=data.column_names,
        num_proc=args.workers or None,
        fn_kwargs={
            "tokenizer": tokenizer,
            "pairs": {(pair["concept"], pair["antagonist"]): position for position, pair in enumerate(pairs)},
        },
        desc="tokenizing",
    )
    lengths = np.asarray(data["length"], dtype=np.int64)
    log.info(f"stories: {len(data)}, {lengths.sum() / 1e6:.1f}M tokens, median {int(np.median(lengths))}")

    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for position in np.argsort(lengths, kind="stable"):
        length = int(lengths[position])
        candidate = max(longest, length)
        if current and ((len(current) + 1) * candidate > args.token_budget or len(current) >= 256):
            batches.append(current)
            current, longest, candidate = [], 0, length
        current.append(int(position))
        longest = candidate
    if current:
        batches.append(current)
    padded = sum(len(batch) * int(lengths[batch].max()) for batch in batches)
    log.info(f"batches: {len(batches)}, padding waste {100 * (padded - lengths.sum()) / max(padded, 1):.2f}%")

    dtype = (
        getattr(torch, args.dtype)
        if args.dtype != "auto"
        # including_emulation=False matters: a V100 reports bf16 as supported but emulates it in
        # software, where it measures 9x slower than fp16 and slower even than fp32.
        else (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
    )
    config.num_hidden_layers = max(layers)
    log.info(f"model: building {max(layers)} of {depth} blocks, dropping the top {depth - max(layers)}")
    model = AutoModel.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": "cuda"},
    )
    model.norm = torch.nn.Identity()  # we read the pre-norm residual stream
    model.eval()
    log.info(f"model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, {dtype}")

    pooling: list[torch.Tensor] = [torch.zeros(0)]
    captured: dict[int, torch.Tensor] = {}
    for position, layer in enumerate(layers):
        model.layers[layer - 1].register_forward_hook(
            partial(capture, position=position, pooling=pooling, captured=captured)
        )

    groups = len(pairs) * 4
    sums = torch.zeros(len(layers), groups, hidden, dtype=torch.float32, device="cuda")
    corpus_sum = torch.zeros(len(layers), hidden, dtype=torch.float32, device="cuda")
    moments = torch.zeros(2, len(layers), hidden, hidden, dtype=torch.float64, device="cuda")
    counts = torch.zeros(groups, dtype=torch.int64, device="cuda")
    tokens = torch.zeros(groups, dtype=torch.float64, device="cuda")
    hits = torch.zeros(groups, dtype=torch.float64, device="cuda")
    dropped = torch.zeros(groups, dtype=torch.int64, device="cuda")
    short = torch.zeros(groups, dtype=torch.int64, device="cuda")

    loader = DataLoader(
        data,
        batch_sampler=batches,
        collate_fn=partial(
            collate, prefix=prefix, suffix=suffix, pad=tokenizer.pad_token_id or 0, skip=args.skip_tokens
        ),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
    )

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            pooling[0] = batch["weights"].cuda(non_blocking=True)
            captured.clear()
            model(
                input_ids=batch["ids"].cuda(non_blocking=True),
                attention_mask=batch["mask"].cuda(non_blocking=True),
                use_cache=False,
            )
            pooled = torch.stack([captured[position] for position in range(len(layers))])  # [layer, row, hidden]
            good = torch.isfinite(pooled).all(dim=2).all(dim=0)
            group = batch["group"].cuda(non_blocking=True)

            live, values = group[good], pooled[:, good, :]
            sums.index_add_(1, live, values)
            corpus_sum += values.sum(dim=1)
            for fold in range(2):
                chosen = values[:, live % 2 == fold, :].double()
                moments[fold].baddbmm_(chosen.transpose(1, 2), chosen)
            counts += torch.bincount(live, minlength=groups)
            tokens += torch.bincount(live, weights=batch["length"].cuda()[good].double(), minlength=groups)
            hits += torch.bincount(live, weights=batch["label_hit"].cuda()[good].double(), minlength=groups)
            dropped += torch.bincount(group[~good], minlength=groups)
            short += torch.bincount(group[batch["short"].cuda()], minlength=groups)

            if step % 50 == 0 or step + 1 == len(batches):
                log.info(f"step {step + 1}/{len(batches)}")

    summary = {
        "stories": int(counts.sum()),
        "tokens": int(tokens.sum()),
        "nonfinite": int(dropped.sum()),
        "clamped": int(short.sum()),
        "empty_groups": int((counts == 0).sum()),
        "padding_waste": round(float(padded - lengths.sum()) / max(padded, 1), 5),
    }
    log.info(f"shard {args.shard} done: {json.dumps(summary)}")
    if summary["nonfinite"]:
        log.warning(f"{summary['nonfinite']} stories dropped for inf/nan; if more than a handful use --dtype float32")

    fingerprint = {
        "model": args.model,
        "skip_tokens": args.skip_tokens,
    }
    save_file(
        {
            "sums": sums.reshape(len(layers), len(pairs), 2, 2, hidden).cpu().numpy(),
            "corpus_sum": corpus_sum.cpu().numpy(),
            "moments": moments.cpu().numpy(),
            "counts": counts.reshape(len(pairs), 2, 2).cpu().numpy(),
            "tokens": tokens.cpu().numpy(),
            "hits": hits.cpu().numpy(),
            "dropped": dropped.cpu().numpy(),
            "short": short.cpu().numpy(),
        },
        str(args.out),
        metadata={
            "manifest": json.dumps(
                {
                    **fingerprint,
                    "config_hash": sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
                    "hidden_size": hidden,
                    "n_model_layers": depth,
                    "layers": layers,
                    "dtype": str(dtype).removeprefix("torch."),
                    "rendered_prefix": "",
                    "shard": args.shard,
                    "shards": args.shards,
                    "n_pairs": len(pairs),
                    "axes": {
                        "sums": ["layer", "pair", "side", "fold", "hidden"],
                        "moments": ["fold", "layer", "hidden", "hidden"],
                        "sides": ["concept", "antagonist"],
                    },
                    "summary": summary,
                }
            )
        },
    )
    log.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="shard file to write")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layers", nargs="+", default=["0.3", "0.4", "0.5", "0.6", "0.7"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--skip-tokens", type=int, default=50, help="leading story tokens the mean pool drops")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--token-budget", type=int, default=32768, help="padded tokens per forward pass")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    main(parser.parse_args())
