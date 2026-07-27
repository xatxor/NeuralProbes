#! /usr/bin/env python

import json
import logging
import re
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

log = logging.getLogger("translate")


def resolve(name: str, dtype: torch.dtype) -> Any:
    """Load a translator whatever architecture family it belongs to.

    :param name: Hugging Face model id.
    :param dtype: compute dtype, bf16 where the card supports it natively.

    :return: the loaded model, on the GPU and in eval mode.
    """
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = loader.from_pretrained(name, dtype=dtype, device_map={"": "cuda"})
        except (ValueError, KeyError, OSError) as error:
            log.info(f"{loader.__name__} declined {name}: {error}")
            continue
        model.eval()
        return model
    raise SystemExit(f"no loader accepted {name}")


def ask(model: Any, tokenizer: Any, prompt: str, new_tokens: int) -> str:
    """Send one prompt and return the reply, decoded greedily.

    :param model: the translator, on the GPU in eval mode.
    :param tokenizer: its tokenizer.
    :param prompt: the fully rendered instruction.
    :param new_tokens: cap on the reply length.

    :return: the decoded continuation.
    """
    try:
        batch = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}]],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")
    except TypeError:
        batch = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}]],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
    with torch.inference_mode():
        sequence = model.generate(
            **batch,
            max_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return str(tokenizer.decode(sequence[0, batch["input_ids"].shape[1] :], skip_special_tokens=True))


def numbered(reply: str, expected: int) -> dict[int, str]:
    """Parse a numbered reply back into a mapping, keeping only lines that parse.

    :param reply: the model's output.
    :param expected: how many lines were asked for.

    :return: line number to translated text, for the lines that came back well-formed.
    """
    found = {}
    for line in reply.splitlines():
        match = re.match(r"^\s*(\d+)[.)]\s*(.+?)\s*$", line)
        if match and 1 <= int(match.group(1)) <= expected:
            found[int(match.group(1))] = match.group(2)
    return found


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    # The two poles go in one request so the translation of a contrast stays a contrast: rendering
    # "epistemic humility" and "overconfidence" in separate calls invites two unrelated word choices
    # that no longer read as opposites.
    PAIRS = """Переведи на русский язык эти пары противоположных поведенческих понятий из \
    интерпретируемости языковых моделей. Каждая строка имеет вид "номер. понятие || противоположность".

    Верни ровно столько же строк, в том же формате и с теми же номерами, без пояснений и без кавычек. \
    Держи формулировки короткими, как термины, а не как предложения. Внутри пары выбирай слова так, \
    чтобы они оставались противоположностями друг другу.

    {body}"""

    CLASSES = """Переведи на русский язык эти названия категорий из онтологии поведенческих понятий. \
    Каждая строка имеет вид "номер. название".

    Верни ровно столько же строк, в том же формате и с теми же номерами, без пояснений и без кавычек. \
    Держи названия короткими, как заголовки разделов.

    {body}"""

    labels = pq.read_table(args.pairs, columns=["pair", "concept", "antagonist", "class_name"]).to_pydict()
    total = len(labels["pair"])
    classes = sorted(set(labels["class_name"]))
    log.info(f"{total} concept pairs, {len(classes)} ontology classes")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    log.info(f"{torch.cuda.get_device_name(0)}, using {dtype}, translator {args.model}")
    model = resolve(args.model, dtype)

    concepts: dict[str, dict[str, str]] = {}
    for start in range(0, total, args.batch):
        rows = list(range(start, min(start + args.batch, total)))
        body = "\n".join(
            f"{offset + 1}. {labels['concept'][row]} || {labels['antagonist'][row]}" for offset, row in enumerate(rows)
        )
        parsed = numbered(ask(model, tokenizer, PAIRS.format(body=body), 1400), len(rows))
        for offset, row in enumerate(rows):
            text = parsed.get(offset + 1, "")
            if "||" not in text:
                continue
            left, right = (part.strip() for part in text.split("||", 1))
            if left and right:
                concepts[str(labels["pair"][row])] = {"concept": left, "antagonist": right}
        if start % (args.batch * 10) == 0 or start + args.batch >= total:
            log.info(
                f"{min(start + args.batch, total)}/{total} pairs, {len(concepts)} translated, "
                f"{(time.monotonic() - started) / 60:.1f}m"
            )

    groups: dict[str, str] = {}
    for start in range(0, len(classes), args.batch):
        batch = classes[start : start + args.batch]
        body = "\n".join(f"{offset + 1}. {name}" for offset, name in enumerate(batch))
        parsed = numbered(ask(model, tokenizer, CLASSES.format(body=body), 1400), len(batch))
        for offset, name in enumerate(batch):
            if translated := parsed.get(offset + 1, "").strip():
                groups[name] = translated
    log.info(f"{len(groups)}/{len(classes)} classes translated")

    args.out.write_text(
        json.dumps(
            {"ru": {"concepts": concepts, "classes": groups}, "model": args.model},
            ensure_ascii=False,
            indent=None,
        )
    )
    log.info(
        f"wrote {args.out}: {len(concepts)}/{total} pairs and {len(groups)}/{len(classes)} classes "
        f"in {(time.monotonic() - started) / 60:.1f}m"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("probes-lda/pairs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("scope/data/translations.json"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--batch", type=int, default=16, help="lines per request")
    main(parser.parse_args())
