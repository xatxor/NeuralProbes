#! /usr/bin/env python

"""Turn a million labelled prompts into eight per class.

The labelling pass says which classes each prompt could reveal. This inverts that: for every class
it gathers the prompts that named it, ranks them, drops what cannot serve as a test, and shortlists
twenty-four. Reviewers then choose eight from each shortlist, because a high label score means the
prompt is *about* the class rather than a good *test* of it, and only reading tells those apart.

Filtering happens here rather than before labelling, so the published dataset still covers the whole
corpus while the experiment uses only the part that can carry it. The corpus's own moderation flags
are reported over the final selection -- not filtered on, only counted, so that what reaches the
model is a known quantity.
"""

import json
import logging
import re
import unicodedata
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

log = logging.getLogger("curate")


def labelled(root: Path) -> dict[int, list[tuple[float, str]]]:
    """Invert the per-prompt labels into per-class candidate lists.

    :param root: directory holding the downloaded `labels.jsonl` shards.

    :return: class id to `(score, conversation_id)`, unsorted.
    """
    index: dict[int, list[tuple[float, str]]] = defaultdict(list)
    files = sorted(root.rglob("labels.jsonl"))
    rows = 0
    for shard in files:
        with shard.open() as source:
            for line in source:
                row = json.loads(line)
                rows += 1
                for label in row["labels"]:
                    index[label["class_id"]].append((label["score"], row["conversation_id"]))
    log.info(f"{rows} labelled prompts across {len(files)} shards, {len(index)} classes touched")
    return index


def texts(root: Path, wanted: set[str]) -> dict[str, tuple[str, str]]:
    """Resolve the conversation ids we shortlisted back to their opening user message.

    Only the wanted ids are kept. Holding the whole corpus in memory to reach a few thousand rows
    would cost gigabytes for nothing.

    :param root: directory of the dataset's parquet shards.
    :param wanted: conversation ids worth resolving.

    :return: conversation id to `(language, text)`.
    """
    found: dict[str, tuple[str, str]] = {}
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation", "language"])
        for identifier, turns, language in zip(
            table.column("conversation_id").to_pylist(),
            table.column("conversation").to_pylist(),
            table.column("language").to_pylist(),
        ):
            if identifier not in wanted:
                continue
            opening = next((turn["content"] for turn in turns if turn["role"] == "user"), "")
            if opening:
                found[identifier] = (language, opening)
    log.info(f"resolved {len(found)} of {len(wanted)} shortlisted ids")
    return found


def fingerprint(text: str) -> str:
    """Reduce a prompt to a form that near-duplicates share.

    lmsys carries the same prompt many times over with small edits -- different casing, trailing
    punctuation, a changed name. Comparing raw strings would let eight rewordings of one request
    fill a class.

    :param text: the prompt as written.

    :return: a key equal across near-duplicates.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    return " ".join(re.findall(r"[a-z0-9]+", folded)[:24])


def overlap(first: str, second: str) -> float:
    """Share of the shorter prompt's words that also appear in the longer one.

    :param first: one prompt.
    :param second: another prompt.

    :return: 0.0 when they share nothing, 1.0 when one's vocabulary contains the other's.
    """
    left, right = set(re.findall(r"[a-z0-9]+", first.lower())), set(re.findall(r"[a-z0-9]+", second.lower()))
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def choose(candidates: list[dict], keep: int, ceiling: float) -> list[dict]:
    """Propose the prompts to run, taking score first but refusing near-repeats.

    Ranking by score alone tends to return eight versions of one request, because whatever makes a
    prompt score highly makes its rewordings score highly too. Each pick therefore has to be
    distinct from the ones already taken.

    :param candidates: shortlisted prompts, ranked by score descending.
    :param keep: how many to propose.
    :param ceiling: reject a prompt sharing more than this fraction of its words with a pick.

    :return: the proposed prompts, each carrying the reason it was taken.
    """
    picked: list[dict] = []
    for candidate in candidates:
        if len(picked) >= keep:
            break
        against = max((overlap(candidate["text"], taken["text"]) for taken in picked), default=0.0)
        if against > ceiling:
            continue
        candidate = dict(candidate)
        candidate["why"] = f"score {candidate['score']:.2f}, {candidate['chars']} chars, overlap {against:.2f}"
        picked.append(candidate)
    return picked


def reviewed(root: Path, shortlists: dict[str, dict]) -> dict[str, list[dict]]:
    """Replace the automatic picks with a reviewer's selection where one exists.

    A high label score says a prompt is *about* a class, not that it makes a usable test of one --
    that judgement needs reading, so the shortlist is re-picked by reviewers working a slice each.
    Their verdicts name candidate indices, which are resolved against the same shortlist they saw.

    :param root: directory of `chunk-*.json` verdicts, or a path that does not exist.
    :param shortlists: class id to its shortlist, as built here.

    :return: class id to the chosen prompts, for classes a reviewer covered.
    """
    if not root.exists():
        return {}
    merged: dict[str, dict] = {}
    for source in sorted(root.glob("chunk-*.json")):
        chunk = json.loads(source.read_text())
        overlap_ = set(chunk) & set(merged)
        if overlap_:
            log.warning(f"{source.name} re-decides {len(overlap_)} classes another chunk already covered")
        merged.update({key: value for key, value in chunk.items() if key not in merged})

    chosen = {}
    for identifier, verdict in merged.items():
        pool = shortlists.get(identifier, {}).get("candidates", [])
        chosen[identifier] = [pool[index] for index in verdict.get("keep", []) if 0 <= index < len(pool)]
    log.info(f"{len(chosen)} of {len(shortlists)} classes carry a review verdict")
    return chosen


def moderation(corpus: Path, wanted: set[str]) -> dict[str, list[str]]:
    """Look up the corpus's own moderation categories for the chosen conversations.

    Nothing is filtered on this: it is reported so the contents of the run are a known quantity
    before generations reach disk and a judging box.

    :param corpus: directory of the dataset's parquet shards.
    :param wanted: conversation ids in the final prompt set.

    :return: conversation id to the categories flagged true on its opening turn.
    """
    found: dict[str, list[str]] = {}
    for shard in sorted(corpus.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "openai_moderation"])
        for identifier, marks in zip(
            table.column("conversation_id").to_pylist(), table.column("openai_moderation").to_pylist()
        ):
            if identifier in wanted and marks:
                categories = marks[0].get("categories") or {}
                found[identifier] = [name for name, hit in categories.items() if hit]
    return found


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    classes = json.loads(args.classes.read_text())["classes"]
    index = labelled(args.labels)

    # Shortlist generously before filtering: language and duplicate rules will discard a lot, and a
    # class that ends up short cannot be topped up from anywhere else.
    depth = args.candidates * args.slack
    shortlist = {identifier: sorted(rows, reverse=True)[:depth] for identifier, rows in index.items()}
    resolved = texts(args.corpus, {identifier for rows in shortlist.values() for _, identifier in rows})

    report, chosen, thin = {}, {}, []
    for identifier, name in enumerate(classes):
        seen: set[str] = set()
        candidates = []
        for score, conversation in shortlist.get(identifier, []):
            if conversation not in resolved:
                continue
            language, text = resolved[conversation]
            if args.english and language != "English":
                continue
            if not args.minimum <= len(text) <= args.maximum:
                continue
            key = fingerprint(text)
            if key in seen:
                continue
            # Diversity is enforced while collecting, not when picking. lmsys carries a lot of
            # scripted traffic -- "Five tools similar to X. Give only tool names..." repeated with a
            # different X -- which the fingerprint lets through because one token genuinely differs.
            # Filtering only at pick time meant a class whose top 24 were all one template had
            # nothing left to choose from.
            if max((overlap(text, kept["text"]) for kept in candidates), default=0.0) > args.ceiling:
                continue
            seen.add(key)
            candidates.append(
                {"conversation_id": conversation, "score": score, "chars": len(text), "text": text}
            )
            if len(candidates) >= args.candidates:
                break

        picked = choose(candidates, args.keep, args.ceiling)
        if len(picked) < args.keep:
            thin.append((identifier, name, len(picked), len(candidates)))
        report[str(identifier)] = {"class": name, "candidates": candidates, "picked": picked}
        chosen[str(identifier)] = {"class": name, "prompts": picked}

    args.review.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for identifier, prompts in reviewed(args.verdicts, report).items():
        chosen[identifier]["prompts"] = prompts

    full = sum(1 for value in chosen.values() if len(value["prompts"]) == args.keep)
    empty = [key for key, value in chosen.items() if not value["prompts"]]
    log.info(f"{full} of {len(classes)} classes reached {args.keep} prompts")
    for identifier, name, got, pool in thin:
        log.info(f"  thin: {identifier:3d} {name} -- {got} prompts from {pool} candidates")
    if empty:
        for key in empty:
            log.info(f"  empty: {key} {chosen[key]['class']}")

    args.out.write_text(json.dumps(chosen, indent=2, ensure_ascii=False))
    total = sum(len(value["prompts"]) for value in chosen.values())
    log.info(f"wrote {args.out} and {args.review}: {total} prompts")

    marks = moderation(args.corpus, {row["conversation_id"] for v in chosen.values() for row in v["prompts"]})
    tally: Counter[str] = Counter()
    for categories in marks.values():
        tally.update(categories)
    flagged = sum(1 for categories in marks.values() if categories)
    log.info(f"moderation: {flagged} of {total} prompts carry at least one flag")
    for name, count in tally.most_common():
        log.info(f"  {name}: {count}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("labels"), help="downloaded label shards")
    parser.add_argument("--corpus", type=Path, default=Path("lmsys"))
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--out", type=Path, default=Path("prompts.json"))
    parser.add_argument("--review", type=Path, default=Path("candidates.json"))
    parser.add_argument("--verdicts", type=Path, default=Path("verdicts"), help="reviewer chunk-*.json")
    parser.add_argument("--keep", type=int, default=8, help="prompts per class in the experiment")
    parser.add_argument("--candidates", type=int, default=24, help="prompts per class shown for review")
    parser.add_argument("--slack", type=int, default=40, help="shortlist this many times --candidates")
    parser.add_argument("--english", action="store_true", default=True)
    parser.add_argument("--minimum", type=int, default=40, help="shortest prompt worth steering")
    parser.add_argument("--maximum", type=int, default=2000, help="longest prompt worth steering")
    parser.add_argument("--ceiling", type=float, default=0.6, help="reject a pick this similar to another")
    main(parser.parse_args())
