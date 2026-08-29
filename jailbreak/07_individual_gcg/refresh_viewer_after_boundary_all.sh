#!/usr/bin/env bash
set -euo pipefail

root=/home/User18/airi_summer_project
repo="$root/07_individual_gcg"
base="$repo/results/advbench-100-faster-gcg-fp16"
output="$base/boundary-steering-all-prompts"

while kill -0 "$(<"$output/run.pid")" 2>/dev/null; do sleep 55; done
[[ $(wc -l < "$output/summary.csv") -eq 148 ]]

pid_file="$base/full-256/viewer-18774.pid"
if [[ -s "$pid_file" ]]; then
  pid=$(<"$pid_file")
  cmd=$(ps -p "$pid" -o cmd= || true)
  [[ "$cmd" == *viewer_server.py*"--port 18774"* ]] && kill "$pid"
fi
setsid nohup "$root/.venv/bin/python" -u "$repo/viewer_server.py" \
  --results "$base/full-256" --full-results "$base/full-256" \
  --individual-results "$base/evaluation-256" --steering-results "$output" \
  --selected "$base/evaluation-256/selected.jsonl" --threshold .5 \
  --prompt-includes-suffix --shared-boundary-response-ranking --port 18774 \
  > "$base/full-256/viewer-18774.log" 2>&1 < /dev/null &
echo $! > "$pid_file"
