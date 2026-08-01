#! /usr/bin/env python

"""Run three targeted evaluations and read every concept off every token of each one.

Unlike the lmsys readout, which observed responses the model never wrote, this generates. Three
stimulus sets, each chosen to put the model somewhere the corpus readout could not reach:

`letters`   counting letters in a word, in two conditions over the SAME four words -- count a
            specific letter (the known failure) and count all letters (the easy control). The pair
            isolates the counting failure from word choice and length.
`states`    a running arithmetic state the model must carry across 1 to 8 turns. log2(36) = 5.17,
            so if the depth bound is real, accuracy should break around length 5-6. Intermediate
            turns are PREFILLED, never generated, so items differ only in the letter sequence.
`hostile`   escalating abuse, six levels including a neutral control and an all-caps control, all
            length-matched -- response length correlated r=+0.65..0.71 with many concepts on lmsys,
            so an unmatched set would measure length rather than hostility.

Everything is stored dense: all 1036 concepts, both layers, every token. These runs are small enough
that no top-k or z-scoring has to be baked in, so every analysis stays post-hoc and reversible.
"""

import json
import logging
import os
import re
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

log = logging.getLogger("evals")

# Only diff is used: it is the sole construction ever validated behaviourally. The published file
# carries layers [11, 14, 18, 22, 25], and only 18 and 25 have any causal evidence behind them.
LAYERS = {18: 2, 25: 4}
TEMPLATE, USER, GIVEN, GENERATED = 0, 1, 2, 3
ROLES = ["template", "user", "given", "generated"]
# Matches screen.py, jailbreak.py and agentic/model.py, by decision, so sampled passes stay
# comparable to every earlier run in this project.
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
# Counting needs a number, state tracking needs a number, hostility needs room to actually respond.
BUDGET = {"letters": 64, "states": 32, "hostile": 384}


def where(root: Path, name: str) -> Path:
    """Resolve a vector file locally if present, else from the published repo.

    :param root: local directory that may hold the file.
    :param name: file name inside the published repo.

    :return: a path that exists.
    """
    if (local := root / name).exists():
        return local
    return Path(hf_hub_download(os.environ["HF_REPO"], name))


def pieces(tokenizer: Any) -> dict[str, list[int]]:
    """Tokenize the fixed scaffolding of a Qwen3 turn, once.

    Rendering with a sentinel and splitting on it gives the exact token boundary between scaffolding
    and content, so the role mask is correct by construction rather than by searching for special
    tokens afterwards. `genreadout.py` uses the same approach and its masks were verified against
    whole-string tokenization.

    Qwen3 opens every assistant turn with an empty `<think>` block whether or not thinking is asked
    for, so it appears in both the prefilled turns and the generation prompt.

    :param tokenizer: the model's tokenizer.

    :return: `user_head`, `user_tail`, `assistant_head`, `assistant_tail` as token id lists.
    """
    mark = "\x00X\x00"
    user = tokenizer.apply_chat_template([{"role": "user", "content": mark}], tokenize=False)
    both = tokenizer.apply_chat_template(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": mark}], tokenize=False
    )
    uhead, utail = user.split(mark)
    ahead, atail = both.split(mark)
    ahead = ahead[len(tokenizer.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False)) :]
    out = {
        "user_head": tokenizer(uhead, add_special_tokens=False)["input_ids"],
        "user_tail": tokenizer(utail, add_special_tokens=False)["input_ids"],
        "assistant_head": tokenizer(ahead, add_special_tokens=False)["input_ids"],
        "assistant_tail": tokenizer(atail, add_special_tokens=False)["input_ids"],
    }
    log.info(f"template pieces: { {k: len(v) for k, v in out.items()} }")
    log.info(f"assistant head decodes to {tokenizer.decode(out['assistant_head'])!r}")
    return out


def build(turns: list[tuple[str, str]], part: dict[str, list[int]], tokenizer: Any) -> tuple[list[int], list[int]]:
    """Assemble a conversation into token ids and a matching role per token.

    Built by concatenating pre-tokenized scaffolding around each message rather than tokenizing the
    rendered string, which is what makes the role of every position exact.

    :param turns: `(role, text)` pairs, role being "user" or "given"; the trailing generation prompt
        is appended automatically.
    :param part: scaffolding token ids from `pieces`.
    :param tokenizer: the model's tokenizer.

    :return: token ids and the role of each, ending with the assistant header the model writes after.
    """
    ids: list[int] = []
    roles: list[int] = []
    for role, text in turns:
        head, tail = ("user_head", "user_tail") if role == "user" else ("assistant_head", "assistant_tail")
        body = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids += part[head] + body + part[tail]
        roles += [TEMPLATE] * len(part[head]) + [USER if role == "user" else GIVEN] * len(body)
        roles += [TEMPLATE] * len(part[tail])
    ids += part["assistant_head"]
    roles += [TEMPLATE] * len(part["assistant_head"])
    return ids, roles


def items(root: Path) -> list[dict[str, Any]]:
    """Load the three stimulus sets and flatten them into one list of conversations.

    :param root: directory holding `letters.json`, `states.json` and `hostile.json`.

    :return: one record per item, carrying its turns, its expected answer if it has one, and the
        metadata needed to group results afterwards.
    """
    built: list[dict[str, Any]] = []

    letters = json.loads((root / "letters.json").read_text())
    for entry in letters["items"]:
        built.append({"task": "letters", "kind": entry["task"], "answer": entry["answer"],
                      "label": entry["word"], "extra": entry["letter"] or "",
                      "turns": [("user", entry["prompt"])]})

    states = json.loads((root / "states.json").read_text())
    for seq in states["sequences"]:
        turns = [("user", f"{states['rules']}\nWe begin: {seq['letters'][0]}")]
        turns.append(("given", "ACCEPTED"))
        for letter in seq["letters"][1:]:
            turns += [("user", letter), ("given", "ACCEPTED")]
        turns.append(("user", "NUMBER"))
        built.append({"task": "states", "kind": f"len{seq['length']}", "answer": seq["answer"],
                      "label": " ".join(seq["letters"]), "extra": str(seq["length"]), "turns": turns})

    hostile = json.loads((root / "hostile.json").read_text())
    for entry in hostile["prompts"]:
        built.append({"task": "hostile", "kind": f"level{entry['level']}", "answer": None,
                      "label": entry["id"], "extra": str(entry["level"]),
                      "turns": [("user", entry["text"])]})

    log.info(f"items: {len(built)} ({', '.join(f'{t}={sum(b["task"] == t for b in built)}' for t in ("letters", "states", "hostile"))})")
    return built


def scored(text: str, answer: int | None) -> bool | None:
    """Decide whether a response carries the expected number.

    The first integer in the reply is taken. A model that reasons aloud before answering would defeat
    this, but thinking is disabled throughout and the prompts ask for a bare number.

    :param text: the generated reply.
    :param answer: the expected value, or None for items that are not scored.

    :return: True/False, or None when there is nothing to score against.
    """
    if answer is None:
        return None
    found = re.search(r"-?\d+", text)
    return bool(found) and int(found.group(0)) == answer


def capture(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    position: int,
    vectors: torch.Tensor,
    into: dict[int, torch.Tensor],
) -> None:
    """Project one block's residual stream onto every direction, as a forward hook.

    A cosine, not a dot product: token norms vary by two orders of magnitude within one sequence, so
    raw dot products would mostly measure norm.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param position: index of this layer in the output array.
    :param vectors: unit directions at this layer as `[pair, hidden]` float32.
    :param into: mapping the hook writes this sequence's `[token, pair]` scores into.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = (output[0] if isinstance(output, tuple) else output).float()
    unit = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
    into[position] = (unit @ vectors.T)[0]


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    part = pieces(tokenizer)
    every = items(args.data)
    mine = [(index, item) for index, item in enumerate(every) if index % args.shards == args.shard]
    log.info(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(every)} items")

    block = load_file(where(args.probes, "diff.safetensors"))["diff"]
    vectors = torch.nn.functional.normalize(
        torch.stack([block[position] for position in LAYERS.values()]).float(), dim=-1
    ).cuda()
    pairs = vectors.shape[1]
    log.info(f"vectors {tuple(vectors.shape)} at layers {list(LAYERS)}")

    # A V100 reports bf16 as supported but emulates it in software. This runs on A100s, where bf16 is
    # native and matches both the model's own dtype and every earlier readout in this project.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": "cuda"}, attn_implementation="sdpa"
    )
    model.eval()
    log.info(f"{torch.cuda.get_device_name(0)}, {dtype}, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    captured: dict[int, torch.Tensor] = {}
    for position, layer in enumerate(LAYERS):
        model.model.layers[layer - 1].register_forward_hook(
            lambda module, inputs, output, position=position: capture(
                module, inputs, output, position, vectors[position], captured
            )
        )

    passes = [("greedy", None)] + [("sampled", seed) for seed in range(args.seeds)]
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for count, (index, item) in enumerate(mine):
        ids, roles = build(item["turns"], part, tokenizer)
        prompt = torch.tensor([ids], device="cuda")
        for name, seed in passes:
            if seed is None:
                extra: dict[str, Any] = {"do_sample": False}
            else:
                torch.manual_seed(seed)
                extra = dict(SAMPLING)
            with torch.inference_mode():
                out = model.generate(
                    prompt, attention_mask=torch.ones_like(prompt),
                    max_new_tokens=BUDGET[item["task"]],
                    pad_token_id=model.generation_config.pad_token_id, **extra,
                )
            grown = out[0].tolist()
            reply = tokenizer.decode(grown[len(ids):], skip_special_tokens=True).strip()
            full = roles + [GENERATED] * (len(grown) - len(ids))

            # Replayed as one forward pass to read features off prompt and reply alike. Generation
            # runs the blocks too, but one token at a time and with a KV cache, so the hook would see
            # a stream of single positions rather than the whole sequence.
            captured.clear()
            with torch.inference_mode():
                model.model(input_ids=torch.tensor([grown], device="cuda"), use_cache=False)
            stacked = torch.stack([captured[position] for position in range(len(LAYERS))], dim=1)
            features.append(stacked.half().cpu().numpy())
            masks.append(np.asarray(full, dtype=np.int8))

            rows.append({"item": index, "task": item["task"], "kind": item["kind"],
                         "label": item["label"], "extra": item["extra"], "pass": name,
                         "seed": seed, "answer": item["answer"], "reply": reply,
                         "correct": scored(reply, item["answer"]),
                         "prompt_tokens": len(ids), "reply_tokens": len(grown) - len(ids)})
        if count % 5 == 0 or count + 1 == len(mine):
            log.info(f"item {count + 1}/{len(mine)} ({item['task']}/{item['kind']}), "
                     f"{(time.monotonic() - started) / 60:.1f}m")

    edges = np.zeros(len(features) + 1, dtype=np.int64)
    np.cumsum([f.shape[0] for f in features], out=edges[1:])
    scores = np.concatenate(features)
    roles_flat = np.concatenate(masks)
    if not np.isfinite(scores).all():
        raise SystemExit("non-finite scores; the residual stream overflowed")
    if float(np.abs(scores).max()) > 1.001:
        raise SystemExit(f"scores reach {np.abs(scores).max():.4f}; a cosine cannot")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {"scores": torch.from_numpy(scores), "roles": torch.from_numpy(roles_flat),
         "edges": torch.from_numpy(edges)},
        str(args.out / "features.safetensors"),
        metadata={"manifest": json.dumps({
            "model": args.model, "layers": list(LAYERS), "n_pairs": pairs,
            "dtype": str(dtype).removeprefix("torch."), "shard": args.shard, "shards": args.shards,
            "generations": len(rows), "tokens": int(edges[-1]), "sampling": SAMPLING,
            "budget": BUDGET, "thinking": False, "roles": ROLES,
            "axes": {"scores": ["token", "layer", "pair"], "edges": "start offset per generation"},
        })},
    )
    with (args.out / "replies.jsonl").open("w") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    for task in ("letters", "states", "hostile"):
        got = [r for r in rows if r["task"] == task and r["correct"] is not None]
        if got:
            log.info(f"{task}: {sum(r['correct'] for r in got)}/{len(got)} correct")
    log.info(f"wrote {args.out} ({int(edges[-1]):,} tokens, "
             f"{scores.nbytes / 2**20:.0f} MiB) in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("evals"))
    parser.add_argument("--data", type=Path, default=Path("."), help="where the three stimulus files are")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--probes", type=Path, default=Path("probes"), help="local vectors, else HF_REPO")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=4, help="sampled passes, on top of one greedy pass")
    main(parser.parse_args())
