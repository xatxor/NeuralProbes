# NeuralProbes

Residual-stream statistics for behavioral concept vectors,
extracted from `Qwen/Qwen3-8B` over [`AntonKorznikov/feature_stories`](https://huggingface.co/datasets/AntonKorznikov/feature_stories) dataset.

## Script reference
| Name | Description |
| - | - |
| [genstats.py](./genstats.py) | Sweep the corpus and accumulate sums, counts and second moments |
| [genvectors.py](./genvectors.py) | Merge the shards into `diff`, `concept_centered`, `antagonist_centered` and `lda` |
| [whiten.py](./whiten.py) | Build all six constructions into one `readouts.safetensors`, verified against the published `lda` |
| [sample.py](./sample.py) | Fix the 40-behaviour jailbreak sample from WildJailbreak into `behaviours.json` |
| [screen.py](./screen.py) | Generate the ±α pairs for every published direction, plus 65 controls |
| [judge.py](./judge.py) | Blinded pairwise judging, or per-response outcome grading |
| [hits.py](./hits.py) | Turn judgements into a hit rate against an empirical null |
| [jailbreak.py](./jailbreak.py) | Read every concept off every token of a jailbreak exchange, and steer with `--steer` |
| [blobs.py](./blobs.py) | Z-score the readouts against the benign baseline and pack them for the viewer |
| [translate.py](./translate.py) | Russian names for the concepts and ontology classes |
