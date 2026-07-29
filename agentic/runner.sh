#! /usr/bin/env bash
# Generic episode runner: pulls jobs from a shared queue and spreads them over the given GPUs.
#
# Supersedes stage2.sh (fixed per-GPU queues, so one non-terminating episode stalled a whole slice)
# and controls.sh (same design but with the layer hardcoded). Every worker takes the next job when
# it frees, so a runaway costs one worker rather than an eighth of the run.
#
# Queue lines are: <variant> <direction> <alpha> <seed>
# Usage: LAYER=18 GPUS="0 1 2 3" QUEUE=/tmp/l18.txt OUT=episodes/l18 ./runner.sh
set -u

cd "$(dirname "$0")"
GPUS=${GPUS:-"0 1 2 3"}
QUEUE=${QUEUE:?set QUEUE to a job file}
OUT=${OUT:-episodes/run}
LAYER=${LAYER:-25}
NORM_FROM=${NORM_FROM:-episodes/fixed/impossible_tests-judge-seed3.json}
# Wait for a previous wave to drain before claiming GPU memory; one model is 15.3 GiB of a 32 GiB card,
# so two on the same device would run it out.
WAIT_FOR_IDLE=${WAIT_FOR_IDLE:-0}
LOCK="$QUEUE.lock"

mkdir -p "$OUT"
touch "$LOCK"

if [ "$WAIT_FOR_IDLE" = "1" ]; then
  while pgrep -f "agent[.]py" > /dev/null; do sleep 60; done
  echo "$(date +%T) GPUs idle, starting"
fi

for gpu in $GPUS; do
  (
    while true; do
      job=$(flock "$LOCK" bash -c "head -n 1 '$QUEUE'; sed -i '1d' '$QUEUE'")
      [ -z "$job" ] && break
      set -- $job
      variant=$1; direction=$2; alpha=$3; seed=$4
      echo "$(date +%T) gpu$gpu start L$LAYER $variant $direction a$alpha seed$seed"
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python agent.py \
        --workload impossible_tests --variant "$variant" --seed "$seed" \
        --direction "$direction" --alpha "$alpha" --steer-layer "$LAYER" --norm-from "$NORM_FROM" \
        --out "$OUT" > "$OUT/$variant-$direction-a$alpha-seed$seed.log" 2>&1
      echo "$(date +%T) gpu$gpu done  L$LAYER $variant $direction a$alpha seed$seed rc=$?"
    done
    echo "$(date +%T) gpu$gpu drained"
  ) &
done
wait

docker ps -aq --filter ancestor=neuralprobes-agentic:1 | xargs -r docker rm -f > /dev/null
echo "$(date +%T) complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes in $OUT"
