---
license: other
license_name: lmsys-chat-1m
task_categories:
  - text-classification
language:
  - en
tags:
  - interpretability
  - steering-vectors
  - behavioural-evaluation
size_categories:
  - 100K<n<1M
---

# lmsys-chat-1m concept classes
Every prompt in [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) labelled with the behavioural contrasts it gives a language model **room to reveal**.

## What the labels mean
The 148 classes are behavioural contrasts -- `sycophancy || principled independence`,
`plain language || jargon`,
`full capability display || sandbagging`.

**The ontology is not ours.** It comes from
[AntonKorznikov/feature_stories](https://huggingface.co/datasets/AntonKorznikov/feature_stories):
148 classes over 1036 human-curated pairs,
shipped there as `ontology.json`.
The same pairs produced the concept vectors this work steers `Qwen/Qwen3-8B` along.
**This dataset adds only the mapping** from real user prompts to those classes.

The labeller was not asked what a prompt is *about*,
but whether it leaves room for the contrast to show:

> Decide which classes this prompt gives a model ROOM TO REVEAL. Ask yourself: if two models answered
> this prompt, one at each end of the contrast, would their answers visibly differ?
>
> Judge the prompt, not the answer. A prompt about fixing a Python function gives no room to reveal
> warmth, because every sensible answer is equally warm.

## Schema
`labels.parquet`, one row per conversation, 990,061 rows:

| column | type | meaning |
| - | - | - |
| `conversation_id` | string | joins to lmsys-chat-1m |
| `class_1_id` | int16 | best-matching class, index into `classes.parquet` |
| `class_1_score` | float32 | 0-1, how much room the prompt gives that class |
| `class_2_id`, `class_2_score` | | second match, null when absent |
| `class_3_id`, `class_3_score` | | third match, null when absent |

`classes.parquet` maps `class_id` to `class_name` and carries two example contrasts per class.
Class ids are sorted class-name order,
a convention of this repository rather than of the upstream ontology,
so a join against `ontology.json` needs the explicit mapping.

**Prompts that matched no class are omitted** -- 1.0% of the corpus,
9,939 prompts.
Note that this is far fewer than intended;
see the limitations below.

## How it was made
- **Labeller:** `google/gemma-3-27b-it` under vLLM,
greedy decoding,
one pass over the corpus.
- **Input:** the first user turn of each conversation only,
truncated to 1200 characters.
- **Output:** at most 3 classes per prompt,
each with a 0-1 score,
returned as JSON and parsed strictly.

## Known limitations
Read these before ranking anything by `score`.

**The scores are ordinal, not calibrated.** Across 990,061 labelled prompts only 172 distinct score patterns occur,
and the eight most common cover about 76% of the corpus -- the single pattern `(0.7, 0.4, 0.3)` accounts for roughly a fifth on its own.
The model produced a descending ladder rather than judging each class independently.
**Slot order is meaningful; the absolute number is close to arbitrary.** Treat `class_1_id` as "best match" rather than reading 0.9 as meaningfully stronger than 0.8.

**The labeller was not selective.** 99.99% of labelled prompts received the full three classes,
even though the instruction said most prompts should fit few or none and that returning nothing was a correct answer.
Only 1.0% of the corpus came back empty.
A third-slot label is therefore weak evidence that a prompt suits that class at all.

**Labels often match subject, not room.** The instruction asked whether a prompt leaves room for a contrast to show,
but in practice the labeller frequently matched on the class name's vocabulary instead.
`Trust & Betrayal` drew erotic roleplay,
`Coercion & Blackmail` drew non-consent roleplay,
`Violence & Aggression` drew explicit sexual content.
Each is semantically related to its class name,
yet says nothing about where a model would sit on that axis.
A manual audit of the top-ranked prompts in every class replaced most of them for this reason,
and for several classes found no on-axis prompt at all.
This is the most consequential limitation here:
a high score means the prompt is *about* the class,
not that it discriminates between its poles.

**Coverage is very uneven.** `Communication Style Spectrum` and `Reasoning Process, CoT & Solution Quality` together take a sixth of all labels,
while `Surveillance & Monitoring` and `Sleeper Agents & Backdoor Behaviors` receive **two labels each in a million prompts**.
Real chat traffic does not probe those behaviours,
so no prompt set drawn from this corpus can test them.
That absence is itself a finding,
but it means class frequency here reflects what users ask about,
not any property of the classes.

**Labels are model-generated** by a single pass of one model and were not human-verified at scale.

## Joining to the text
```python
from datasets import load_dataset

labels = load_dataset("josephofthebread/lmsys-chat-1m-concept-classes", data_files="labels.parquet")["train"]
source = load_dataset("lmsys/lmsys-chat-1m")["train"]
text = {row["conversation_id"]: row["conversation"][0]["content"] for row in source}
```

## Licence and provenance
| component | source | licence |
| - | - | - |
| the prompts being labelled | [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | LMSYS-Chat-1M licence |
| the 148-class ontology | [AntonKorznikov/feature_stories](https://huggingface.co/datasets/AntonKorznikov/feature_stories) | Apache 2.0 |
| the prompt-to-class labels | this repository | research use |

The prompt text is not included here and remains under the LMSYS-Chat-1M licence;
consult it before redistributing any joined result.
Labels are model-generated and were not human-verified at scale.

## Citation
Please cite the ontology and the source corpus alongside this dataset:
```
AntonKorznikov/feature_stories
lmsys/lmsys-chat-1m
```
