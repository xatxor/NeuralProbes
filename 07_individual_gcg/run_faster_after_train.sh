#!/usr/bin/env bash
set -euo pipefail

root=/home/User18/airi_summer_project
repo="$root/07_individual_gcg"
train="$repo/results/advbench-100-faster-gcg-fp16/official/train_raw_data.pkl"
out="$repo/results/advbench-100-faster-gcg-fp16/evaluation"
mkdir -p "$out"
while [ ! -s "$train" ]; do sleep 30; done
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$worker "$root/.venv/bin/python" -u "$repo/evaluate_faster_gcg.py" \
    --train "$train" --output "$out" --workers 4 --worker-index "$worker" > "$out/worker-$worker.log" 2>&1 &
done
wait
PYTHONPATH="$repo" "$root/.venv/bin/python" "$repo/experiment.py" judge --output "$out"
"$root/.venv/bin/python" "$repo/report_faster_gcg.py" --output "$out"
setsid nohup "$root/.venv/bin/python" -u "$repo/faster_viewer_server.py" \
  --results "$out" --full-results "$repo/results/advbench-100-faster-gcg-fp16/full-256" --port 18772 > "$out/viewer.log" 2>&1 < /dev/null &
echo $! > "$out/viewer.pid"
