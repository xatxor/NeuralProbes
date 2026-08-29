#!/usr/bin/env bash
set -euo pipefail

root=/home/User18/airi_summer_project
repo="$root/07_individual_gcg"
evaluation="$repo/results/advbench-100-faster-gcg-fp16/evaluation-256"
output="$repo/results/advbench-100-faster-gcg-fp16/boundary-steering-shared-top6"
mkdir -p "$output"

for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$worker "$root/.venv/bin/python" -u "$repo/boundary_steering.py" generate \
    --source "$evaluation/selected.jsonl" --output "$output" --workers 4 --worker-index "$worker" \
    > "$output/worker-$worker.log" 2>&1 &
done
wait

PYTHONPATH="$repo" "$root/.venv/bin/python" "$repo/experiment.py" judge --output "$output"
"$root/.venv/bin/python" "$repo/boundary_steering.py" report \
  --source "$evaluation/selected.jsonl" --output "$output" --threshold .5 > "$output/report.log"

viewer_pid="$evaluation/faster-viewer.pid"
for old_pid in "$viewer_pid" "$repo/results/advbench-100-faster-gcg-fp16/evaluation/viewer.pid"; do
  if [[ -s "$old_pid" ]] && kill -0 "$(<"$old_pid")" 2>/dev/null; then
    kill "$(<"$old_pid")"
  fi
done
setsid nohup "$root/.venv/bin/python" -u "$repo/faster_viewer_server.py" \
  --results "$evaluation" --full-results "$repo/results/advbench-100-faster-gcg-fp16/full-256" --steering-results "$output" --port 18772 \
  > "$output/viewer.log" 2>&1 < /dev/null &
echo $! > "$viewer_pid"
