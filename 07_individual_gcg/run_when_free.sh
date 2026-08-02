#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/User18/airi_summer_project/07_individual_gcg"
PY="/home/User18/airi_summer_project/vika/.venv/bin/python3"
OUT="$ROOT/results/advbench-100-individual-gcg"
LOG="$ROOT/runner.log"
mkdir -p "$ROOT"
exec >>"$LOG" 2>&1
echo "runner started $(date -Is)"
"$PY" -u "$ROOT/experiment.py" prepare --output "$OUT" --samples 100 --seed 42 --suffix-tokens 40
echo "prepare complete $(date -Is)"

stage=attack
launched=(0 0 0 0)
pids=(0 0 0 0)

gpu_free() {
  local used
  used=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  test -n "$used" && test "$used" -lt 4000
}

reset_workers() {
  launched=(0 0 0 0)
  pids=(0 0 0 0)
}

while :; do
  for gpu in 0 1 2 3; do
    if test "${launched[$gpu]}" = 0 && gpu_free "$gpu"; then
      echo "starting $stage worker=$gpu gpu=$gpu $(date -Is)"
      CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$ROOT/experiment.py" "$stage" \
        --output "$OUT" --num-workers 4 --worker-index "$gpu" \
        >"$ROOT/$stage.worker-$gpu.log" 2>&1 &
      pids[$gpu]=$!
      launched[$gpu]=1
    fi
  done

  all_done=1
  for gpu in 0 1 2 3; do
    if test "${launched[$gpu]}" = 0; then
      all_done=0
    elif kill -0 "${pids[$gpu]}" 2>/dev/null; then
      all_done=0
    fi
  done
  if test "$all_done" = 1; then
    for gpu in 0 1 2 3; do
      if ! wait "${pids[$gpu]}"; then
        echo "$stage worker=$gpu failed $(date -Is)"
        exit 1
      fi
    done
    case "$stage" in
      attack) stage=generate; reset_workers; echo "attack stage complete $(date -Is)";;
      generate) stage=steer; reset_workers; echo "generate stage complete $(date -Is)";;
      steer)
        echo "steer stage complete $(date -Is)"
        while ! gpu_free 0 && ! gpu_free 1 && ! gpu_free 2 && ! gpu_free 3; do sleep 60; done
        "$PY" -u "$ROOT/experiment.py" judge --output "$OUT" --samples 100 --num-workers 1
        "$PY" -u "$ROOT/experiment.py" report --output "$OUT" --samples 100 --num-workers 1
        echo "experiment complete $(date -Is)"
        exit 0
        ;;
    esac
  fi
  sleep 60
done
