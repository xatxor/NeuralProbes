# Probes

Residual-stream statistics for behavioral concept vectors,
extracted from `Qwen/Qwen3-8B` over [`AntonKorznikov/feature_stories`](https://huggingface.co/datasets/AntonKorznikov/feature_stories) dataset.

Зависимости и `uv` — в корне репозитория. Запуск из корня:

```bash
uv sync --group vectors
uv run probes/genstats.py --out probes/shard_0.safetensors --shard 0 --shards 8
uv run probes/genvectors.py probes/shard_*.safetensors --out probes/artifacts/
```
