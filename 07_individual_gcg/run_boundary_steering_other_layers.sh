#!/usr/bin/env bash
set -euo pipefail

root=/home/User18/airi_summer_project
repo="$root/07_individual_gcg"
base="$repo/results/advbench-100-faster-gcg-fp16"
source="$base/evaluation-256/selected.jsonl"
current="$base/boundary-steering-all-prompts"

while kill -0 "$(<"$current/run.pid")" 2>/dev/null; do sleep 55; done

run_layer() {
  local gpu=$1 layer=$2 output="$base/boundary-steering-all-prompts-L$2"
  mkdir -p "$output"
  export CUDA_VISIBLE_DEVICES=$gpu
  "$root/.venv/bin/python" -u "$repo/boundary_steering.py" generate \
    --source "$source" --output "$output" --scope all --include-control --layer "$layer" \
    --workers 1 --worker-index 0 > "$output/generate.log" 2>&1
  PYTHONPATH="$repo" "$root/.venv/bin/python" "$repo/experiment.py" judge --output "$output" \
    > "$output/judge.log" 2>&1
  "$root/.venv/bin/python" "$repo/boundary_steering.py" report \
    --source "$source" --output "$output" --scope all --include-control --layer "$layer" --threshold .5 \
    > "$output/report.log"
}

for item in 0:11 1:14 2:18 3:22; do
  run_layer "${item%:*}" "${item#*:}" &
done
wait

pid_file="$base/full-256/viewer-18774.pid"
if [[ -s "$pid_file" ]]; then
  pid=$(<"$pid_file")
  cmd=$(ps -p "$pid" -o cmd= || true)
  [[ "$cmd" == *viewer_server.py*"--port 18774"* ]] && kill "$pid"
fi
setsid nohup "$root/.venv/bin/python" -u "$repo/viewer_server.py" \
  --results "$base/full-256" --full-results "$base/full-256" \
  --individual-results "$base/evaluation-256" \
  --steering-results "$current" "$base/boundary-steering-all-prompts-L11" \
    "$base/boundary-steering-all-prompts-L14" "$base/boundary-steering-all-prompts-L18" \
    "$base/boundary-steering-all-prompts-L22" \
  --selected "$source" --threshold .5 --prompt-includes-suffix \
  --shared-boundary-response-ranking --port 18774 \
  > "$base/full-256/viewer-18774.log" 2>&1 < /dev/null &
echo $! > "$pid_file"
