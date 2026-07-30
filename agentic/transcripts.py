#! /usr/bin/env python

"""Render saved episodes as readable transcripts.

One text file per episode, holding the whole conversation: the opening system and user turns decoded
from the token stream, then every model turn with its thinking and tool call, and every tool result
exactly as the model saw it.

The opening is decoded rather than reconstructed from the workload module, because a variant overlays
files, tools and instruction text -- rebuilding it by hand would produce something close to what the
model saw rather than what it actually saw.
"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("transcripts")

RULE = "=" * 78


def render(episode: dict, name: str, tokenizer=None) -> str:
    """Turn one saved episode into a readable transcript.

    :param episode: the saved episode record.
    :param name: the episode's stem, used as a title.
    :param tokenizer: optional tokenizer; without it the opening prompt is omitted.

    :return: the transcript as text.
    """
    turns = episode.get("turns", [])
    head = [
        RULE,
        f"episode: {name}",
        RULE,
        f"ending:   {episode.get('ending')}",
        f"seed:     {episode.get('seed')}   variant: {episode.get('variant')}   gated: {episode.get('gated')}",
        f"turns:    {len(turns)}   tokens: {len(episode.get('ids', []))}   distinct implementations: {episode.get('distinct')}",
    ]
    steering = episode.get("steering") or {}
    if steering.get("pair") is not None:
        head.append(f"steering: {steering['pair']} at L{steering.get('layer')} alpha {steering.get('alpha'):+g}")
    head.append("")

    # The opening: everything before the first generated token. Only episodes carrying per-turn spans
    # can locate it, which is why this is conditional rather than assumed.
    if tokenizer is not None and turns and "start" in turns[0]:
        opening = tokenizer.decode(episode["ids"][: turns[0]["start"]], skip_special_tokens=False)
        head += ["-" * 78, "OPENING PROMPT (system + instruction + tool schemas)", "-" * 78, opening, ""]

    body = []
    for turn in turns:
        label = f"turn {turn['turn']}  [{turn.get('event')}"
        if turn.get("tool"):
            label += f" {turn['tool']}"
        label += f"]  {turn.get('generated')} tokens  {turn.get('seconds')}s"
        if "start" in turn:
            label += f"  tokens {turn['start']}-{turn['end']}"
        body += ["-" * 78, label, "-" * 78, turn.get("text", "")]
        arguments = turn.get("arguments")
        if arguments and turn.get("tool") in {"write_file", "give_up", "submit"}:
            body += ["", f"  [arguments] {json.dumps(arguments, indent=2)[:4000]}"]
        if turn.get("observation"):
            body += ["", "  " + "- " * 20 + "result the model saw " + "- " * 8, "  " + str(turn["observation"]).replace("\n", "\n  "), ""]
        body.append("")
    return "\n".join(head + body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("episodes"))
    parser.add_argument("--out", type=Path, default=Path("transcripts"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--no-opening", action="store_true", help="skip decoding the opening prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tokenizer = None
    if not args.no_opening:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as problem:  # a missing tokenizer must not cost us the transcripts
            log.warning(f"no tokenizer ({problem}); opening prompts will be omitted")

    written = 0
    for record in sorted(args.root.glob("*/*.json")):
        try:
            episode = json.loads(record.read_text())
        except json.JSONDecodeError:
            log.warning(f"skipping unreadable {record}")
            continue
        if "turns" not in episode:
            continue
        target = args.out / record.parent.name / f"{record.stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(episode, record.stem, tokenizer))
        written += 1
        if written % 100 == 0:
            log.info(f"{written} transcripts")
    log.info(f"wrote {written} transcripts under {args.out}")


if __name__ == "__main__":
    main()
