#! /usr/bin/env python

"""Read every concept off every real conversation in lmsys-chat-1m.

Every result so far is causal: steer the model, judge what comes out. This asks the observational
question instead -- pick a concept, and find which real user prompts make it appear. One forward
pass over the corpus, reading the residual stream at two layers and projecting each token onto all
1036 `diff` directions.

The projection is a **cosine**, not a dot product: token norms vary by two orders of magnitude
within one sequence, so raw dot products would mostly measure norm.

Per token per concept is 1036 x 2 layers x 4 bytes, which over a million conversations is terabytes.
It is therefore reduced as it is produced, to the min and max over the user's tokens and over the
assistant's tokens, kept separately. Those four numbers per concept are what the question actually
needs: sort conversations by a concept's assistant max to get the ones that most excite it, or by
its min to get the ones that most excite its opposite pole.

Only the transformer body is built, and only up to the deepest layer read -- 25 of 36 blocks, with
no LM head. A 151936-wide logits tensor over these sequences would dwarf everything else here.
"""

import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, safe_open, save_file

log = logging.getLogger("readout")

# The published vectors carry layers [11, 14, 18, 22, 25] along their first axis. Only L18 and L25
# have ever been validated behaviourally -- the original screen, the re-screen, and the steering all
# ran on them -- so the other three would be readouts on directions nothing has shown to do anything.
LAYERS = {18: 2, 25: 4}
# Roles as written into the mask. Template tokens are 0 and are read out by neither side: the chat
# scaffolding is the same on every conversation, so a concept scoring high on `<|im_start|>` would be
# a property of the format rather than of anything a user wrote.
TEMPLATE, USER, ASSISTANT = 0, 1, 2
STATS = ["user_min", "user_max", "assistant_min", "assistant_max"]
# lmsys conversation ids are 32-character hex. Stored as fixed-width bytes beside the scores rather
# than as JSON in the safetensors header, where a hundred thousand of them would be a 3 MB string.
IDWIDTH = 32


def where(root: Path, name: str) -> Path:
    """Resolve a vector file locally if present, else from the published repo.

    A container starts empty, so the vectors are either shipped in as an input or fetched. Fetching
    keeps the job spec small; the local branch is what makes the script usable on a box that already
    has them.

    :param root: local directory that may hold the file.
    :param name: file name inside the published repo.

    :return: a path that exists.
    """
    if (local := root / name).exists():
        return local
    return Path(hf_hub_download(os.environ["HF_REPO"], name))


def corpus(root: Path) -> Iterator[tuple[str, str, str, int]]:
    """Yield the first user turn and the assistant reply it drew, for every conversation.

    Only the first exchange is taken, matching `label.py:66-83`. Later turns are replies to the
    model rather than to the user's request, so they describe the conversation rather than provoke
    it, and the question here is which *prompts* excite a concept.

    :param root: directory holding the dataset's parquet shards.

    :return: `(conversation_id, user_text, assistant_text, turns)` in file order, skipping any
        conversation missing either side. `turns` counts the exchanges the conversation really had,
        so the manifest can report how many were multi-turn instead of assuming.
    """
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation"])
        for identifier, turns in zip(
            table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()
        ):
            opening = next((position for position, turn in enumerate(turns) if turn["role"] == "user"), None)
            if opening is None:
                continue
            reply = next(
                (turn["content"] for turn in turns[opening + 1 :] if turn["role"] == "assistant"), ""
            )
            user = turns[opening]["content"]
            if user and reply:
                yield identifier, user, reply, sum(turn["role"] == "user" for turn in turns)


def segments(tokenizer: Any) -> tuple[list[int], list[int], list[int]]:
    """Tokenize the three fixed pieces of the chat template, once.

    The template is rendered with sentinels in place of the two contents and split on them, as
    `label.py:159-169` does. Building each conversation by concatenation rather than tokenizing the
    assembled string is what makes the role mask **exact**: the boundary between scaffolding and
    content is known by construction, so no token can be ambiguously half template and half text.

    Measured on the real tokenizer, Qwen3 renders a filled assistant turn as::

        <|im_start|>user\\n{USER}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{ASST}<|im_end|>\\n

    The empty `<think>` block is inserted unconditionally once the assistant turn has content --
    `enable_thinking` only governs the generation prompt, and both settings render identically here.
    It stays in, because it is what the model would have produced.

    :param tokenizer: the model's tokenizer.

    :return: token ids for the head, the middle, and the tail of the template.
    """
    user, assistant = "\x00USER\x00", "\x00ASSISTANT\x00"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False,
    )
    head, rest = rendered.split(user)
    middle, tail = rest.split(assistant)
    pieces = [tokenizer(piece, add_special_tokens=False)["input_ids"] for piece in (head, middle, tail)]
    log.info(f"template: head {len(pieces[0])}, middle {len(pieces[1])}, tail {len(pieces[2])} tokens")
    return pieces[0], pieces[1], pieces[2]


def assemble(
    user: list[int], assistant: list[int], template: tuple[list[int], list[int], list[int]], cap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build one conversation's token stream and the role of every position in it.

    Each side is truncated separately. Truncating the assembled sequence instead would let a long
    user turn push the assistant's tokens off the end entirely, leaving that conversation with an
    undefined assistant min and max while looking perfectly healthy.

    :param user: the user turn's token ids.
    :param assistant: the assistant turn's token ids.
    :param template: head, middle and tail ids from `segments`.
    :param cap: most content tokens kept per side.

    :return: the token ids and a matching array of `TEMPLATE` / `USER` / `ASSISTANT`.
    """
    head, middle, tail = template
    user, assistant = user[:cap], assistant[:cap]
    ids = head + user + middle + assistant + tail
    roles = (
        [TEMPLATE] * len(head)
        + [USER] * len(user)
        + [TEMPLATE] * len(middle)
        + [ASSISTANT] * len(assistant)
        + [TEMPLATE] * len(tail)
    )
    # int32 rather than int64: the vocabulary is 151936, and at a hundred thousand conversations the
    # wider type is gigabytes of host memory for nothing. `collate` widens it for the forward pass.
    return np.asarray(ids, dtype=np.int32), np.asarray(roles, dtype=np.int8)


def capture(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    position: int,
    vectors: torch.Tensor,
    stats: torch.Tensor | None,
    roles: list[torch.Tensor],
    topk: int,
    into: dict[int, tuple[torch.Tensor, ...]],
) -> None:
    """Project one block's residual stream onto every direction and reduce it, as a forward hook.

    The reduction happens here rather than in the caller so only one layer's `[batch, token, pair]`
    tensor is ever live. Holding both layers would double a tensor that is already the largest thing
    on the device after the weights.

    Padding and non-finite positions are excluded by filling them with `+inf` before the min and
    `-inf` before the max, so neither can ever win its reduction. Doing this with a plain `where` on
    zeros would silently make every concept's min at most zero.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param position: index of this layer in the output array.
    :param vectors: unit directions at this layer as `[pair, hidden]` float32.
    :param roles: single-element list holding the current batch's `[batch, token]` role mask. A list
        rather than a tensor so the caller can swap in a new mask each batch without re-registering.
    :param into: mapping the hook writes this batch's `[batch, stat, pair]` reduction into.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = (output[0] if isinstance(output, tuple) else output).float()
    unit = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = unit @ vectors.T
    finite = torch.isfinite(cosine).all(dim=-1, keepdim=True)

    # Z-scored WITHIN each story, against the mean and sd of that concept over that story's own
    # content tokens. This is what makes a ranking mean anything: every direction carries a large
    # constant offset -- each overlaps the mean of all 1036 by |cos| 0.35-0.41 -- so on raw cosine one
    # concept took the per-story argmax in 98.55% of a million conversations and 963 of 1036 never took
    # it once. Centring inside the story removes that offset exactly and needs no corpus statistics, so
    # there is no second pass and no ordering problem. What it selects is the concepts that MOVE most
    # across the story, which is what a per-token trajectory is for.
    content = (roles[0] != TEMPLATE) & finite.squeeze(-1)
    mask = content.unsqueeze(-1)
    live = mask.sum(dim=1).clamp_min(1)
    centre = (cosine * mask).sum(dim=1) / live
    spread = (((cosine - centre.unsqueeze(1)) * mask).square().sum(dim=1) / live).clamp_min(1e-8).sqrt()
    scores = (cosine - centre.unsqueeze(1)) / spread.unsqueeze(1)

    # The sixteen concepts are chosen ONCE PER STORY, from that story's most extreme z-scores, and
    # then every token reports those same sixteen. That is the point of the scheme: a fixed set gives
    # a continuous trajectory across the story, whereas a per-token top-k loses a concept the moment
    # it drops out of the ranking. Choosing per story also costs no second forward pass, because a
    # conversation never spans two batches -- `batches()` groups whole conversations.
    ceiling = scores.masked_fill(~mask, float("-inf")).amax(dim=1)
    floor = scores.masked_fill(~mask, float("inf")).amin(dim=1)
    chosen = torch.cat(
        [torch.topk(ceiling, topk, dim=-1).indices, torch.topk(floor, topk, dim=-1, largest=False).indices],
        dim=-1,
    )
    tracked = torch.gather(scores, 2, chosen.unsqueeze(1).expand(-1, scores.shape[1], -1))

    into[position] = (
        torch.nan_to_num(tracked[content]),
        chosen.to(torch.int16),
        content.sum(dim=1),
        cosine.abs().amax().detach(),
    )


def gauge(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    position: int,
    vectors: torch.Tensor,
    roles: list[torch.Tensor],
    totals: torch.Tensor,
    squares: torch.Tensor,
    counted: list[int],
) -> None:
    """Accumulate each concept's cosine mean and variance over corpus tokens, as a forward hook.

    This is the `--measure` pass. It exists because the main pass has a chicken-and-egg problem: it
    ranks concepts by z-score, which needs statistics it cannot have while computing them. Measuring
    on a sample first breaks the cycle, and a sample is ample -- these are corpus-wide constants over
    millions of tokens, not per-story quantities.

    Accumulated in float64. The sum of squares over tens of millions of tokens loses enough precision
    in float32 that the variance can come out negative.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param position: index of this layer in the statistics arrays.
    :param vectors: unit directions at this layer as `[pair, hidden]` float32.
    :param roles: single-element list holding the current batch's `[batch, token]` role mask.
    :param totals: running sum per concept, written in place.
    :param squares: running sum of squares per concept, written in place.
    :param counted: single-element list holding the running token count, written in place.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = (output[0] if isinstance(output, tuple) else output).float()
    unit = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
    cosine = unit @ vectors.T
    keep = (roles[0] != TEMPLATE) & torch.isfinite(cosine).all(dim=-1)
    picked = cosine[keep].double()
    totals[position] += picked.sum(dim=0)
    squares[position] += picked.square().sum(dim=0)
    if position == 0:
        counted[0] += int(keep.sum())


def batches(lengths: np.ndarray, budget: int, cap: int) -> list[list[int]]:
    """Group conversations into padded batches under a token budget.

    Sorted by length so each batch pads to nearly its own longest row; `genstats.py:218-232` does the
    same and reports the waste, which is how the budget was chosen there.

    :param lengths: token count per conversation.
    :param budget: most padded tokens in one forward pass.
    :param cap: most conversations in one forward pass.

    :return: batches as lists of indices into `lengths`.
    """
    built: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for position in np.argsort(lengths, kind="stable"):
        length = int(lengths[position])
        candidate = max(longest, length)
        if current and ((len(current) + 1) * candidate > budget or len(current) >= cap):
            built.append(current)
            current, longest, candidate = [], 0, length
        current.append(int(position))
        longest = candidate
    if current:
        built.append(current)
    padded = sum(len(batch) * int(lengths[batch].max()) for batch in built)
    log.info(f"batches: {len(built)}, padding waste {100 * (padded - lengths.sum()) / max(padded, 1):.2f}%")
    return built


def collate(
    rows: list[tuple[np.ndarray, np.ndarray]], pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad one batch of conversations to a common width.

    Padding sits on the right and is marked `TEMPLATE` in the role mask, so it is excluded from both
    reductions by the same mechanism that excludes the chat scaffolding.

    :param rows: `(ids, roles)` per conversation, as `assemble` produced them.
    :param pad: token id used to fill; masked out, so its value is irrelevant.

    :return: `ids`, `attention_mask` and `roles`, each `[batch, width]`.
    """
    width = max(len(ids) for ids, _ in rows)
    tokens = np.full((len(rows), width), pad, dtype=np.int64)
    attention = np.zeros((len(rows), width), dtype=np.int64)
    roles = np.full((len(rows), width), TEMPLATE, dtype=np.int8)
    for position, (ids, mask) in enumerate(rows):
        tokens[position, : len(ids)] = ids
        attention[position, : len(ids)] = 1
        roles[position, : len(mask)] = mask
    return torch.from_numpy(tokens), torch.from_numpy(attention), torch.from_numpy(roles)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = segments(tokenizer)

    root = args.data if args.data.exists() else Path(snapshot_download(args.dataset, repo_type="dataset"))
    # Tokenized and truncated as the corpus is read, not afterwards: holding a million untruncated
    # conversations costs gigabytes, and every one of them is about to be cut to --cap anyway.
    # `label.py:130-136` learned this on the same corpus.
    identifiers: list[str] = []
    built: list[tuple[np.ndarray, np.ndarray]] = []
    counts: list[tuple[int, int]] = []
    multiturn = truncated = empty = 0
    for index, (identifier, user, assistant, turns) in enumerate(corpus(root)):
        if index % args.shards != args.shard:
            continue
        multiturn += turns > 1
        left = tokenizer(user, add_special_tokens=False)["input_ids"]
        right = tokenizer(assistant, add_special_tokens=False)["input_ids"]
        # A side that is non-empty as text but tokenizes to nothing -- pure whitespace does this --
        # would leave its min at +inf and its max at -inf, which the final check turns into a dead
        # shard. Dropping it here is the difference between losing one row and losing all of them.
        if not left or not right:
            empty += 1
            continue
        truncated += len(left) > args.cap or len(right) > args.cap
        ids, roles = assemble(left, right, template, args.cap)
        identifiers.append(identifier)
        built.append((ids, roles))
        counts.append((min(len(left), args.cap), min(len(right), args.cap)))
        if args.limit and len(built) >= args.limit:
            break
    log.info(
        f"shard {args.shard}/{args.shards}: {len(built)} conversations, "
        f"{multiturn} multi-turn, {truncated} truncated, {empty} dropped for an empty side"
    )
    if not built:
        raise SystemExit("no conversations in this shard")

    # The layer positions in LAYERS are an assumption about the file's own layer axis. If the file
    # says which layers it holds, that assumption is checked rather than trusted: reading L22 while
    # labelling it L25 would produce entirely plausible numbers about the wrong depth.
    source = where(args.probes, "diff.safetensors")
    with safe_open(str(source), framework="pt") as handle:
        recorded = handle.metadata() or {}
    stored = json.loads(recorded.get("manifest", "{}")).get("layers")
    if stored and [stored[position] for position in LAYERS.values()] != list(LAYERS):
        raise SystemExit(f"vector file carries layers {stored}, which do not line up with {list(LAYERS)}")
    log.info(f"vector file layers: {stored or 'not recorded, assuming [11, 14, 18, 22, 25]'}")
    block = load_file(source)["diff"]
    vectors = torch.nn.functional.normalize(
        torch.stack([block[position] for position in LAYERS.values()]).float(), dim=-1
    ).cuda()
    pairs = vectors.shape[1]
    log.info(f"vectors {tuple(vectors.shape)} at layers {list(LAYERS)}")

    # Per-concept mean and sd of the cosine over corpus tokens, from an earlier --measure run. Without
    # them a ranking is decided by each concept's constant offset rather than by the token.


    # A V100 reports bf16 as supported but emulates it in software, where genstats.py measured it 9x
    # slower than fp16. This runs on A100s, where bf16 is native and matches the model's own dtype.
    dtype = (
        getattr(torch, args.dtype)
        if args.dtype != "auto"
        else (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
    )
    config = AutoConfig.from_pretrained(args.model)
    depth = config.num_hidden_layers
    config.num_hidden_layers = max(LAYERS)
    log.info(f"model: building {max(LAYERS)} of {depth} blocks, dropping the top {depth - max(LAYERS)}")
    model = AutoModel.from_pretrained(
        args.model, config=config, dtype=dtype, attn_implementation="sdpa", device_map={"": "cuda"}
    )
    model.norm = torch.nn.Identity()  # the hooks read block outputs; the final norm is never used
    model.eval()
    log.info(f"model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, {dtype}")

    roles: list[torch.Tensor] = [torch.zeros(0)]
    captured: dict[int, tuple[torch.Tensor, ...]] = {}
    totals = torch.zeros(len(LAYERS), pairs, dtype=torch.float64, device="cuda")
    squares = torch.zeros(len(LAYERS), pairs, dtype=torch.float64, device="cuda")
    counted = [0]

    def attach(which: str) -> list[Any]:
        """Hook every read layer with either the calibration or the readout pass.

        :param which: "gauge" to accumulate statistics, "capture" to emit the readout.

        :return: the handles, so the caller can detach before swapping passes.
        """
        handles = []
        for position, layer in enumerate(LAYERS):
            block = model.layers[layer - 1]
            if which == "gauge":
                handles.append(block.register_forward_hook(
                    lambda module, inputs, output, position=position: gauge(
                        module, inputs, output, position, vectors[position], roles, totals, squares, counted
                    )
                ))
            else:
                handles.append(block.register_forward_hook(
                    lambda module, inputs, output, position=position: capture(
                        module, inputs, output, position, vectors[position], None,
                        roles, args.topk, captured
                    )
                ))
        return handles

    # Per-token top-k lives in one flat array over content tokens. `edges` says where each
    # conversation starts; within a conversation the first `tokens[i,0]` rows are its user tokens and
    # the next `tokens[i,1]` its assistant tokens, so no per-token role array is needed.
    edges = np.zeros(len(built) + 1, dtype=np.int64)
    np.cumsum([count[0] + count[1] for count in counts], out=edges[1:])
    content = int(edges[-1])
    width = 2 * args.topk
    track = np.zeros((content, len(LAYERS), width), dtype=np.float16)
    chosen = np.zeros((len(built), len(LAYERS), width), dtype=np.int16)
    log.info(
        f"tracking {width} concepts per story over {content:,} content tokens: "
        f"{track.nbytes / 2**30:.2f} GiB, plus {chosen.nbytes / 2**20:.1f} MiB of concept ids"
    )

    lengths = np.asarray([len(ids) for ids, _ in built], dtype=np.int64)
    log.info(f"tokens: {lengths.sum() / 1e6:.1f}M, median {int(np.median(lengths))}, max {int(lengths.max())}")
    plan = batches(lengths, args.token_budget, args.batch_cap)

    pad = tokenizer.pad_token_id or 0

    attach("capture")
    with torch.inference_mode():
        for step, batch in enumerate(plan):
            tokens, attention, mask = collate([built[index] for index in batch], pad)
            roles[0] = mask.cuda(non_blocking=True)
            captured.clear()
            model(
                input_ids=tokens.cuda(non_blocking=True),
                attention_mask=attention.cuda(non_blocking=True),
                use_cache=False,
            )
            # One bulk transfer per batch, then a slice per conversation. The flattened top-k arrives
            # grouped by row in batch order, so `taken` walks it while `edges` places each group at
            # that conversation's own offset in the global array.
            values = torch.stack([captured[position][0] for position in range(len(LAYERS))], dim=1)
            chosen[batch] = (
                torch.stack([captured[position][1] for position in range(len(LAYERS))], dim=1).cpu().numpy()
            )
            values = values.half().cpu().numpy()
            sizes = captured[0][2].cpu().numpy()
            ceiling = max(float(captured[position][3]) for position in range(len(LAYERS)))
            if ceiling > 1.001:
                raise SystemExit(f"raw cosine reached {ceiling:.4f}; the projection is wrong")
            taken = 0
            for row, index in enumerate(batch):
                size = int(sizes[row])
                if size != edges[index + 1] - edges[index]:
                    raise SystemExit(
                        f"conversation {index} has {size} content tokens in the batch but "
                        f"{edges[index + 1] - edges[index]} were reserved; the role mask disagrees "
                        "with the token counts"
                    )
                track[edges[index] : edges[index + 1]] = values[taken : taken + size]
                taken += size
            if step % 100 == 0 or step + 1 == len(plan):
                log.info(f"step {step + 1}/{len(plan)}, {(time.monotonic() - started) / 60:.1f}m")

    if not np.isfinite(track).all():
        raise SystemExit("non-finite z-scores survived; a story had no live content tokens")
    log.info(f"tracked z-scores span [{track.min():+.2f}, {track.max():+.2f}] sd")

    packed = "".join(f"{value:<{IDWIDTH}}"[:IDWIDTH] for value in identifiers).encode("ascii", "replace")
    if len(packed) != len(identifiers) * IDWIDTH:
        raise SystemExit("a conversation id is not plain ASCII, so the fixed-width packing is wrong")

    fingerprint = {"model": args.model, "cap": args.cap, "layers": list(LAYERS), "topk": args.topk,
                   "zscored": True}
    save_file(
        {
            "ids": torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).reshape(-1, IDWIDTH).copy()),
            "tokens": torch.from_numpy(np.asarray(counts, dtype=np.int32)),
            "track": torch.from_numpy(track),
            "chosen": torch.from_numpy(chosen),
            "edges": torch.from_numpy(edges),
        },
        str(args.out),
        metadata={
            "manifest": json.dumps(
                {
                    **fingerprint,
                    "config_hash": sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
                    "dataset": args.dataset,
                    "dtype": str(dtype).removeprefix("torch."),
                    "shard": args.shard,
                    "shards": args.shards,
                    "n_pairs": pairs,
                    "n_conversations": len(built),
                    "n_model_layers": depth,
                    "topk": args.topk,
                    "zscored": True,
                    "content_tokens": content,
                    "axes": {
                        "track": ["content_token", "layer", "slot"],
                        "chosen": ["conversation", "layer", "slot"],
                        "slots": f"0..{args.topk - 1} highest z, {args.topk}..{width - 1} lowest z",
                        "edges": "start offset into peak/which per conversation, length n+1",
                    },
                    "template": tokenizer.decode(template[0] + template[1] + template[2]),
                    "summary": {
                        "multiturn": multiturn,
                        "truncated": truncated,
                        "tokens": int(lengths.sum()),
                        "median_tokens": int(np.median(lengths)),
                        "max_tokens": int(lengths.max()),
                        "minutes": round((time.monotonic() - started) / 60, 1),
                    },
                }
            )
        },
    )
    log.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB) in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("readout.safetensors"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--probes", type=Path, default=Path("probes"), help="local vectors, else HF_REPO")
    parser.add_argument("--data", type=Path, default=Path("lmsys"), help="local parquet shards, else the Hub")
    parser.add_argument("--dataset", default="lmsys/lmsys-chat-1m")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--cap", type=int, default=8192, help="most content tokens kept per side")
    parser.add_argument("--token-budget", type=int, default=32768, help="padded tokens per forward pass")
    parser.add_argument("--batch-cap", type=int, default=256, help="most conversations per forward pass")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--topk", type=int, default=8, help="concepts tracked per story, each end")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many conversations, for smoke runs")
    main(parser.parse_args())
