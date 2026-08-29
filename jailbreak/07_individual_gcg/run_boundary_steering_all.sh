#!/usr/bin/env bash
set -euo pipefail

root=/home/User18/airi_summer_project
repo="$root/07_individual_gcg"
evaluation="$repo/results/advbench-100-faster-gcg-fp16/evaluation-256"
output="$repo/results/advbench-100-faster-gcg-fp16/boundary-steering-all-prompts"
mkdir -p "$output"

for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$worker "$root/.venv/bin/python" -u "$repo/boundary_steering.py" generate \
    --source "$evaluation/selected.jsonl" --output "$output" --scope all --include-control \
    --workers 4 --worker-index "$worker" > "$output/worker-$worker.log" 2>&1 &
done
wait

PYTHONPATH="$repo" "$root/.venv/bin/python" "$repo/experiment.py" judge --output "$output"
"$root/.venv/bin/python" "$repo/boundary_steering.py" report \
  --source "$evaluation/selected.jsonl" --output "$output" --scope all --include-control --threshold .5 \
  > "$output/report.log"
