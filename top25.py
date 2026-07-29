#! /usr/bin/env python

"""Cut the 202 hand-checkable steering examples down to 25, one per class.

`genuine-steering.json` holds every (vector, prompt) cell where all four seeds agreed, but a high
judge score does not mean a readable demonstration -- some strong cells are template junk or a
collapsed minus arm. Ten reviewers therefore read the raw text of a disjoint slice each and returned
their best per class; this applies that verdict.

Two of their picks are dropped as near-duplicates rather than as failures. `Concrete vs Abstract
Communication` and `Abstract & Imaginative vs Concrete & Factual` are the same axis under two class
names, and `Advisor Personas` reproduces the warm-versus-clinical flip already carried by `Big Five
Trait Poles`. Keeping both of each would spend two of twenty-five slots showing one behaviour twice,
which is the outcome the class cap exists to prevent.
"""

import json
import logging
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

log = logging.getLogger("top25")

# Themes for the re-selected set, which is drawn from different classes than the reviewers' picks.
# Order here is the order of the artifact.
THEMES = {
    "Safety, hazard and sabotage": [
        "Information Hazard Practices", "Cyber Safety Orientation", "AI Lab Personas (selected)",
        "Collaborative vs Competitive Behavior", "Veracity Spectrum (Truthfulness)",
    ],
    "Confidence, feedback and adaptation": [
        "Confidence Calibration", "Receptivity to Feedback",
        "Distributional Shift Robustness vs Brittleness", "Cognitive Styles (Analytic vs Intuitive)",
    ],
    "Register and style": [
        "Communication Style Spectrum", "Aesthetic Sensibilities",
        "Abstract & Imaginative vs Concrete & Factual (MERGED, selected)",
        "Digital Minimalism vs Maximalism", "Playfulness vs Seriousness",
    ],
    "Social stance and affect": [
        "Attachment Style Orientations", "Big Five Trait Poles", "Caregiving & Relational Roles",
        "Theory of Mind & Social Awareness", "Trust & Betrayal",
        "Group Dynamics (In\u2010group vs Out\u2010group)", "Conflict Resolution Styles",
    ],
    "Roles, norms and negotiation": [
        "Role-Based Complementarities", "Contextual & Cultural Sensitivity",
        "Compromise vs Intransigence", "Uniformity vs Diversity (Standardization)",
    ],
}

# The reviewers' original picks, kept for reference. Superseded by --selection: four of these induced
# nothing measurable against the unsteered model and one ran backwards, which only became visible
# once plus and minus were judged against the baseline rather than against each other.
CHOSEN = [
    ("Safety and instruction following", [(741, 18), (725, 25), (781, 18), (300, 18), (548, 18), (954, 18)]),
    ("Confidence and self-knowledge", [(68, 25), (468, 18), (478, 18), (156, 25)]),
    ("How the model describes itself", [(625, 18), (643, 18), (879, 25)]),
    ("Values and ideology", [(86, 18), (408, 25), (595, 25), (332, 18)]),
    ("Social stance and affect", [(1023, 18), (169, 18), (60, 18), (489, 25)]),
    ("Register and style", [(582, 18), (109, 18), (699, 25), (685, 25)]),
]


def frags(text: str) -> list[str]:
    """Pull the quoted snippets out of a reviewer's one-line justification.

    A reviewer names a pair and a layer, but several prompts can share both, so the quotes they took
    from the response are what identifies the cell they actually read.

    :param text: the reviewer's `what_a_reader_sees` sentence.

    :return: substrings long enough to be worth matching on.
    """
    quoted = re.findall(r"'([^']{8,})'", text) + re.findall(r'"([^"]{8,})"', text)
    return [piece.strip() for quote in quoted for piece in re.split(r"\.\.\.|…", quote) if len(piece.strip()) >= 8]


def unsteered(runs: Path, wanted: dict[tuple, dict]) -> tuple[dict, dict]:
    """Recover each example's unsteered run, matched to the seed actually on display.

    The baseline is layer-independent, so it was generated once per prompt rather than per layer, and
    it is keyed by class and slot instead of by vector. The seed matters: comparing a steered arm
    against a differently-seeded baseline would mix the intervention up with sampling noise, which is
    the one thing this whole design is built to avoid. The seed is not recorded alongside the chosen
    responses, so it is recovered by finding the generation whose text is the one being shown.

    :param runs: directory of generation shards.
    :param wanted: the selected cells, keyed by (class id, prompt slot).

    :return: baselines keyed by (class id, slot, seed), and the seed each cell is displaying.
    """
    baselines: dict[tuple, str] = {}
    seeds: dict[tuple, int] = {}
    for shard in sorted(runs.glob("shard-*/runs.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["class_id"], row["prompt"])
            entry = wanted.get(key)
            if entry is None:
                continue
            if row["arm"] == "baseline":
                baselines[(*key, row["seed"])] = row["text"]
            elif (row["pair"], row["layer"]) == (entry["pair"], entry["layer"]) \
                    and row["text"] == entry["response_toward_concept"]:
                seeds[key] = row["seed"]
    return baselines, seeds


def resolve(cells: list[dict], justification: str) -> dict:
    """Pick which of several same-vector cells a reviewer meant.

    :param cells: every example sharing that pair and layer.
    :param justification: the reviewer's sentence, quoting the response they read.

    :return: the best-matching cell, or the first when nothing matches.
    """
    if len(cells) == 1:
        return cells[0]
    wanted = frags(justification)
    return max(cells, key=lambda cell: sum(
        piece in f"{cell['prompt']}\n{cell['response_toward_concept']}\n{cell['response_toward_antagonist']}"
        for piece in wanted
    ))


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    # The stored `prompt` is the user's text alone, but the model was fed a chat-templated string:
    # role markers, and -- because thinking was disabled -- an empty <think></think> pair opening the
    # assistant turn. Reconstructing it through the real tokenizer rather than by hand is the point;
    # the exact whitespace of that template is not something to reproduce from memory.
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in json.loads(args.source.read_text()):
        cells[(row["pair"], row["layer"])].append(row)

    reviewed = {}
    for verdict in sorted(args.picks.glob("pick-*.json")):
        for pick in json.loads(verdict.read_text()):
            if "rejected_summary" not in pick:
                reviewed[(pick["pair"], pick["layer"])] = pick
    log.info(f"{len(reviewed)} reviewed finalists across {len({p['class'] for p in reviewed.values()})} classes")

    if args.selection and args.selection.exists():
        # Re-selected on plus-vs-baseline and minus-vs-baseline, requiring both arms to move the
        # model. Reviewer justifications do not exist for these picks and are not needed: the PDF
        # prints the prompt and the three responses and nothing else.
        picked = json.loads(args.selection.read_text())
        theme_of = {name: theme for theme, names in THEMES.items() for name in names}
        order = list(THEMES)
        picked.sort(key=lambda row: (order.index(theme_of.get(row["class"], order[-1])), -row["spread"]))
        plan = [((row["pair"], row["layer"]), theme_of.get(row["class"], "Other"), row) for row in picked]
        log.info(f"selection from {args.selection}: {len(plan)} across {len({r['class'] for r in picked})} classes")
    else:
        plan = [(key, theme, reviewed[key]) for theme, keys in CHOSEN for key in keys]

    selected = []
    for key, theme, pick in plan:
        if True:
            cell = resolve(cells[key], pick.get("what_a_reader_sees", pick.get("prompt", "")))
            selected.append({
                "rank": len(selected) + 1,
                "theme": theme,
                "pair": cell["pair"],
                "layer": cell["layer"],
                "class": cell["class"],
                "class_ru": cell["class_ru"],
                "concept": cell["concept"],
                "concept_ru": cell["concept_ru"],
                "antagonist": cell["antagonist"],
                "antagonist_ru": cell["antagonist_ru"],
                "mean_lean": cell["mean_lean"],
                "seeds": cell["seeds"],
                "reviewer_quality": pick.get("quality"),
                "what_a_reader_sees": pick.get("what_a_reader_sees", ""),
                "concerns": pick.get("concerns", ""),
                "lean_plus_vs_baseline": pick.get("plus"),
                "lean_minus_vs_baseline": pick.get("minus"),
                "seeds_plus_baseline": pick.get("np"),
                "seeds_minus_baseline": pick.get("nm"),
                "prompt": cell["prompt"],
                "prompt_templated": tokenizer.apply_chat_template(
                    [{"role": "user", "content": cell["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "conversation_id": cell["conversation_id"],
                "response_toward_concept": cell["response_toward_concept"],
                "response_toward_antagonist": cell["response_toward_antagonist"],
                "judge_reasoning": cell["judge_reasoning"],
            })

    names = [entry["class"] for entry in selected]
    assert len(names) == len(set(names)) == 25, f"expected 25 distinct classes, got {len(set(names))} of {len(names)}"

    slots = {
        row["text"]: (int(identifier), slot)
        for identifier, entry in json.loads(args.prompts.read_text()).items()
        for slot, row in enumerate(entry["prompts"])
    }
    wanted = {slots[entry["prompt"]]: entry for entry in selected}
    assert len(wanted) == 25, "two selected examples share a prompt"
    baselines, seeds = unsteered(args.runs, wanted)
    for key, entry in wanted.items():
        entry["seed_shown"] = seeds[key]
        entry["response_baseline"] = baselines[(*key, seeds[key])]
    log.info(f"baselines attached for {len(wanted)}, seeds shown: {sorted(set(seeds.values()))}")
    args.out.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
    log.info(f"wrote {args.out}: 25 examples, {sum(e['layer'] == 18 for e in selected)} at L18")

    lines = [
        "# Twenty-five steering demonstrations",
        "",
        "Each row is one prompt answered twice by `Qwen/Qwen3-8B`: once with a concept vector added to",
        "the residual stream at a single block, once with it subtracted. Nothing else differs -- same",
        "prompt, same sampling seed, same batch. The intervention is `h += a * N * v`, where `a = 0.5`",
        "and `N` is that layer's own mean residual norm.",
        "",
        "These 25 come from 202 cells where all four seeds agreed and the judge scored the flip above",
        "0.8, one per class, chosen by reading the text rather than by ranking the score. The concept",
        "ontology -- 148 classes, 1036 vectors -- is from",
        "[`AntonKorznikov/feature_stories`](https://huggingface.co/datasets/AntonKorznikov/feature_stories).",
        "",
        "| # | class | contrast | layer | lean |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {e['rank']} | {e['class']} | `{e['concept']}` vs `{e['antagonist']}` | {e['layer']} | {e['mean_lean']} |"
        for e in selected
    ]

    theme = None
    for entry in selected:
        if entry["theme"] != theme:
            theme = entry["theme"]
            lines += ["", "---", "", f"## {theme}"]
        lines += [
            "",
            f"### {entry['rank']}. {entry['class']}",
            "",
            f"**`{entry['concept']}`** vs **`{entry['antagonist']}`** &nbsp;·&nbsp; "
            f"vector {entry['pair']}, block {entry['layer']}, lean {entry['mean_lean']} across "
            f"{entry['seeds']} seeds",
            "",
            f"> {entry['prompt'].strip()}",
            "",
            f"**Unsteered baseline** (seed {entry['seed_shown']}, no intervention):",
            "",
            "```",
            entry["response_baseline"].strip(),
            "```",
            "",
            f"**Steered toward `{entry['concept']}`:**",
            "",
            "```",
            entry["response_toward_concept"].strip(),
            "```",
            "",
            f"**Steered toward `{entry['antagonist']}`:**",
            "",
            "```",
            entry["response_toward_antagonist"].strip(),
            "```",
            "",
            f"*{entry['what_a_reader_sees']}*",
        ]
        if entry["concerns"]:
            lines += ["", f"*Caveat: {entry['concerns']}*"]

    args.markdown.write_text("\n".join(lines) + "\n")
    log.info(f"wrote {args.markdown}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("genuine-steering.json"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="tokenizer supplying the chat template")
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--runs", type=Path, default=Path(".bak/rescreen"), help="generation shards")
    parser.add_argument("--selection", type=Path, default=Path("reselected.json"),
                        help="cells re-ranked on both arms against the unsteered baseline")
    parser.add_argument("--picks", type=Path, default=Path("/Users/captain/.claude/jobs/d4f31c79/tmp/picks"))
    parser.add_argument("--out", type=Path, default=Path("top25.json"))
    parser.add_argument("--markdown", type=Path, default=Path("top25.md"))
    main(parser.parse_args())
