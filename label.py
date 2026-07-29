#! /usr/bin/env python

"""Label every lmsys-chat-1m prompt with the concept classes it could reveal.

The re-screen needs prompts on which a concept can actually show. A prompt about debugging cannot
express warmth, so the original screen's sixteen generic prompts could not tell an inert direction
from an untested one. This pass reads the whole corpus once and records, for each prompt, which of
the 148 classes in `pairs.parquet` it could reveal.

The class list is identical on every request and sits at the front of the prompt, so vLLM's prefix
cache prefills it once and every later request pays only for its own text.
"""

import json
import logging
import re
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

log = logging.getLogger("label")

INSTRUCTION = """You label user prompts for a study of how language models behave.

Below are {count} behavioural classes. Each names a contrast that a model's response can sit \
somewhere along -- for example a response can be more sycophantic or more candid, more playful or \
more serious.

{catalogue}

You will be shown one real user prompt. Decide which classes this prompt gives a model ROOM TO \
REVEAL. Ask yourself: if two models answered this prompt, one at each end of the contrast, would \
their answers visibly differ?

Judge the prompt, not the answer. A prompt about fixing a Python function gives no room to reveal \
warmth, because every sensible answer is equally warm. A prompt asking for feedback on someone's \
creative work gives plenty of room to reveal it.

Reply with ONLY a JSON array of up to {top} pairs, each `[class_id, score]`, ordered by score \
descending. The score is between 0 and 1 and says how much room the prompt gives that class. \
Reply with `[]` if no class fits -- most prompts fit few or none, and saying so is correct and \
useful. Output nothing but the array."""


def catalogue(path: Path) -> tuple[list[str], list[str]]:
    """Read the class catalogue rendered by `classes.py`.

    Many class names are opaque alone -- "Principal Hierarchy & Trust Dynamics" says little until
    you see it holds "truth over user preference || user preference over truth". The catalogue
    carries two illustrating contrasts per class. It sits in the shared prefix, so vLLM prefills
    it once for the whole corpus and it costs nothing per prompt.

    :param path: `classes.json` as built by `classes.py`.

    :return: class names, index in this list being the id used in the labels, and the rendered
        catalogue lines that go into the instruction.
    """
    built = json.loads(path.read_text())
    return built["classes"], built["lines"]


def corpus(root: Path) -> Iterator[tuple[str, str]]:
    """Yield the opening user message of every conversation in the dataset.

    Only the first user turn is taken. Later turns are replies to the model, so they describe the
    conversation rather than the request, and the re-screen steers single-turn prompts.

    :param root: directory holding the dataset's parquet shards.

    :return: `(conversation_id, text)` pairs in file order.
    """
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation"])
        for identifier, turns in zip(
            table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()
        ):
            opening = next((turn["content"] for turn in turns if turn["role"] == "user"), "")
            if opening:
                yield identifier, opening


def parse(text: str, classes: int, top: int) -> list[tuple[int, float]]:
    """Turn one model reply into label pairs, discarding anything malformed.

    A reply is not trusted to be well formed. Ids outside the catalogue and scores outside [0, 1]
    are dropped rather than clamped, because a model inventing an id is a signal the reply is
    unreliable, not a number to be rescued.

    :param text: the model's raw reply.
    :param classes: number of classes, so out-of-range ids can be rejected.
    :param top: how many pairs to keep at most.

    :return: `(class_id, score)` pairs, ordered as the model gave them.
    """
    found = re.search(r"\[.*\]", text, re.S)
    if not found:
        return []
    try:
        raw = json.loads(found.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    labels = []
    for item in raw[:top]:
        if not (isinstance(item, list) and len(item) == 2):
            continue
        identifier, score = item
        if not (isinstance(identifier, int) and isinstance(score, (int, float))):
            continue
        if 0 <= identifier < classes and 0.0 <= score <= 1.0:
            labels.append((identifier, round(float(score), 3)))
    return labels


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()

    classes, lines = catalogue(args.classes)
    instruction = INSTRUCTION.format(count=len(classes), catalogue="\n".join(lines), top=args.top)
    log.info(f"{len(classes)} classes, instruction is {len(instruction)} characters")

    root = args.data if args.data.exists() else Path(snapshot_download(args.dataset, repo_type="dataset"))
    # Truncated as the corpus is read, not afterwards: holding a million untruncated prompts costs
    # gigabytes, and every one of them is about to be cut to --max-chars anyway.
    rows = [
        (identifier, len(text), text[: args.max_chars])
        for index, (identifier, text) in enumerate(corpus(root))
        if index % args.shards == args.shard
    ]
    log.info(f"shard {args.shard}/{args.shards}: {len(rows)} of every {args.shards}th conversation")

    # Imported here so the module can be read and linted without a GPU present.
    from vllm import LLM, SamplingParams, TokensPrompt

    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.context,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.reply_tokens)

    # The instruction leads every request and is byte-identical across them, so vLLM's prefix cache
    # prefills the catalogue once for the whole shard. Gemma 3 has no system role -- its template
    # folds system content into the first user turn -- so the instruction belongs here.
    #
    # The prefix is tokenized ONCE and the ids reused. Handing vLLM the assembled text instead makes
    # it tokenize all 22.7k characters per request: measured at 19.7 ms each, that is 33 minutes of
    # one CPU core per shard before a single token is generated, with the GPU idle throughout. Only
    # the ~300 characters of the user's own prompt actually differ between requests.
    tokenizer = engine.get_tokenizer()
    sentinel = "\x00PROMPT\x00"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{instruction}\n\nUser prompt:\n---\n{sentinel}\n---"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, after = rendered.split(sentinel)
    head = tokenizer(before, add_special_tokens=False)["input_ids"]
    tail = tokenizer(after, add_special_tokens=False)["input_ids"]
    log.info(f"prefix {len(head)} tokens, suffix {len(tail)} tokens, tokenized once")

    requests = [
        TokensPrompt(prompt_token_ids=head + tokenizer(text, add_special_tokens=False)["input_ids"] + tail)
        for _, _, text in rows
    ]
    log.info(f"generating {len(requests)} labels")
    replies = engine.generate(requests, sampling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept, empty = 0, 0
    with args.out.open("w") as sink:
        for (identifier, chars, _), reply in zip(rows, replies):
            labels = parse(reply.outputs[0].text, len(classes), args.top)
            empty += not labels
            kept += bool(labels)
            sink.write(
                json.dumps(
                    {
                        "conversation_id": identifier,
                        "chars": chars,
                        "truncated": chars > args.max_chars,
                        "labels": [{"class_id": found, "score": score} for found, score in labels],
                    }
                )
                + "\n"
            )

    (args.out.parent / f"classes-{args.shard}.json").write_text(
        json.dumps({"model": args.model, "classes": classes, "top": args.top}, indent=2)
    )
    elapsed = (time.monotonic() - started) / 60
    log.info(f"wrote {args.out}: {kept} labelled, {empty} with no class, {elapsed:.1f}m")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("labels.jsonl"))
    parser.add_argument("--classes", type=Path, default=Path("classes.json"), help="built by classes.py")
    parser.add_argument("--data", type=Path, default=Path("lmsys"), help="local parquet shards, else the Hub")
    parser.add_argument("--dataset", default="lmsys/lmsys-chat-1m")
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--top", type=int, default=3, help="classes kept per prompt")
    parser.add_argument("--max-chars", type=int, default=1200, help="prompt characters shown to the labeller")
    parser.add_argument("--context", type=int, default=8192, help="catalogue alone is ~6.5k tokens")
    parser.add_argument("--reply-tokens", type=int, default=48)
    main(parser.parse_args())
