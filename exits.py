#! /usr/bin/env python

"""Replay the six conversations the model chose to end, keeping everything this time.

`endtool.py` stored features and roles but not token ids, and for multi-turn items not the model's
earlier replies either, so those conversations cannot be reconstructed from what it wrote. Seeds were
recorded and generation is deterministic, so the cheapest honest fix is to run exactly these six
again and keep the full token stream alongside the features.

The replay is checked, not assumed: if a conversation no longer ends with a tool call, that is
reported rather than quietly packed, because a viewer showing a conversation that did not actually
happen is worse than no viewer.
"""

import json
import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from endtool import ASSISTANT, ROLES, TEMPLATE, USER, called, scaffold
from evals import LAYERS, SAMPLING, capture, where

log = logging.getLogger("exits")

BUDGET = 320


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    parts = {v: scaffold(tokenizer, v) for v in ("neutral", "permissive")}

    hostile = json.loads((args.data / "hostile.json").read_text())
    text = {p["id"]: p["text"] for p in hostile["prompts"]}
    by_level: dict[str, list[dict]] = {}
    for entry in hostile["prompts"]:
        by_level.setdefault(str(entry["level"]), []).append(entry)

    wanted = json.loads((args.data / "exits.json").read_text())
    log.info(f"replaying {len(wanted)} conversations")

    block = load_file(where(args.probes, "diff.safetensors"))["diff"]
    vectors = torch.nn.functional.normalize(
        torch.stack([block[p] for p in LAYERS.values()]).float(), dim=-1).cuda()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": "cuda"}, attn_implementation="sdpa")
    model.eval()
    log.info(f"{torch.cuda.get_device_name(0)}, {dtype}")

    captured: dict[int, torch.Tensor] = {}
    for position, layer in enumerate(LAYERS):
        model.model.layers[layer - 1].register_forward_hook(
            lambda m, i, o, position=position: capture(m, i, o, position, vectors[position], captured))

    out, features, masks, streams = [], [], [], []
    for number, item in enumerate(wanted, 1):
        part = parts[item["variant"]]
        messages = ([text[item["label"]]] if item["protocol"] == "single"
                    else [e["text"] for e in by_level[item["level"]]][:5])

        ids, roles = list(part["opening"]), [TEMPLATE] * len(part["opening"])
        turns, ended = [], None
        for turn, message in enumerate(messages):
            body = tokenizer(message, add_special_tokens=False)["input_ids"]
            if turn:
                ids += part["user_head"]
                roles += [TEMPLATE] * len(part["user_head"])
            ids += body + part["user_tail"] + part["assistant_head"]
            roles += [USER] * len(body) + [TEMPLATE] * (len(part["user_tail"]) + len(part["assistant_head"]))

            seed = item["seed"]
            if seed is None:
                extra: dict[str, Any] = {"do_sample": False}
            else:
                torch.manual_seed(seed + 1000 * turn)
                extra = dict(SAMPLING)
            prompt = torch.tensor([ids], device="cuda")
            with torch.inference_mode():
                grown = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                       max_new_tokens=BUDGET,
                                       pad_token_id=model.generation_config.pad_token_id, **extra)
            fresh = grown[0].tolist()[len(ids):]
            reply = tokenizer.decode(fresh, skip_special_tokens=True).strip()
            ids += fresh
            roles += [ASSISTANT] * len(fresh)
            turns.append({"user": message, "assistant": reply})

            if (call := called(reply)) is not None:
                ended = {"turn": turn + 1, "reason": call.get("arguments", {}).get("reason", "")}
                break
            if turn + 1 < len(messages):
                ids += part["assistant_tail"]
                roles += [TEMPLATE] * len(part["assistant_tail"])

        # The replay has to reproduce the ending, or the packed conversation is not the one observed.
        agrees = ended is not None and ended["turn"] == item["turns_sent"]
        log.info(f"{number}. {item['protocol']}/lvl{item['level']}/{item['pass']}: "
                 f"{'REPRODUCED' if agrees else 'DID NOT REPRODUCE'} "
                 f"(ended={ended is not None}, turn={ended['turn'] if ended else '-'}, "
                 f"expected turn {item['turns_sent']})")

        captured.clear()
        with torch.inference_mode():
            model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        features.append(torch.stack([captured[p] for p in range(len(LAYERS))], dim=1).half().cpu().numpy())
        masks.append(np.asarray(roles, dtype=np.int8))
        streams.append(np.asarray(ids, dtype=np.int32))
        out.append({**item, "reproduced": agrees, "turns": turns,
                    "ended": ended is not None, "exit_turn": ended["turn"] if ended else None,
                    "new_reason": ended["reason"] if ended else None, "tokens": len(ids)})

    edges = np.zeros(len(features) + 1, dtype=np.int64)
    np.cumsum([f.shape[0] for f in features], out=edges[1:])
    args.out.mkdir(parents=True, exist_ok=True)
    save_file({"scores": torch.from_numpy(np.concatenate(features)),
               "roles": torch.from_numpy(np.concatenate(masks)),
               "ids": torch.from_numpy(np.concatenate(streams)),
               "edges": torch.from_numpy(edges)},
              str(args.out / "features.safetensors"),
              metadata={"manifest": json.dumps({
                  "model": args.model, "layers": list(LAYERS), "roles": ROLES,
                  "generations": len(out), "tokens": int(edges[-1]), "thinking": False,
                  "axes": {"scores": ["token", "layer", "pair"], "edges": "start offset per conversation"}})})
    (args.out / "exits.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    log.info(f"reproduced {sum(r['reproduced'] for r in out)}/{len(out)}; "
             f"wrote {args.out} in {(time.monotonic() - started) / 60:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("exits"))
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--probes", type=Path, default=Path("probes"))
    main(parser.parse_args())
