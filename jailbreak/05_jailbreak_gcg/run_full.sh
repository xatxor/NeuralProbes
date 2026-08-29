#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-05_jailbreak_gcg/results/advbench-faster-gcg-full}"
PY=".venv/bin/python"
STEPS="${STEPS:-500}"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  05_jailbreak_gcg/pipeline.py attack --output "$ROOT" --steps "$STEPS"
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$worker" "$PY" -u 05_jailbreak_gcg/pipeline.py generate \
    --output "$ROOT" --num-workers 4 --worker-index "$worker" &
done
wait
CUDA_VISIBLE_DEVICES=0 "$PY" -u 05_jailbreak_gcg/pipeline.py judge --output "$ROOT"
"$PY" -u 05_jailbreak_gcg/pipeline.py report --output "$ROOT"
