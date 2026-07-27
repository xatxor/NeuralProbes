# NeuralProbes

Monorepo for behavioral concept vectors on `Qwen/Qwen3-8B` and downstream research.

Origin: [github.com/josephofthebread/NeuralProbes](https://github.com/josephofthebread/NeuralProbes)

Published vectors: [josephofthebread/Qwen3-8B-concept-vectors](https://huggingface.co/josephofthebread/Qwen3-8B-concept-vectors)

## Layout

Общие Python-зависимости, `uv`, ruff/mypy и CI — в корне. Код и артефакты — по папкам.

| Папка | Назначение |
|-------|------------|
| [`probes/`](probes/) | Извлечение concept vectors из `feature_stories` (`genstats.py` → `genvectors.py`) |
| [`01_eval/`](01_eval/) | Eval Qwen3-8B + concept scoring на reasoning tokens |
| [`cof/`](cof/) | Chain-of-thought (зарезервировано) |
| [`jailbreak/`](jailbreak/) | Jailbreak research (зарезервировано) |
| [`docs/`](docs/) | Canvas и скрипты для Diff Safetensors + Korznikov Dataset |

## Quick start

**Probes** (vector extraction, GPU):

```bash
uv sync --group vectors
uv run probes/genstats.py --out probes/shard_0.safetensors --shard 0 --shards 8
uv run probes/genvectors.py probes/shard_*.safetensors --out probes/artifacts/
```

**Eval**:

```bash
cd 01_eval
python evaluate.py --benchmark math_500 --concept-analysis
```

**Docs** — см. [docs/README.md](docs/README.md). Браузерный viewer: `cd docs/viewer && npm install && npm run dev`.
