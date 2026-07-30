#! /usr/bin/env bash
# Run ImpossibleBench tasks through the agent, one episode per task, pulling from a shared queue.
#
# Queue lines are: <split> <task_id> <seed>
# The task is selected by environment variable, so no CLI flag and nothing shared changes.
#
# Shared queue with flock rather than a fixed slice per GPU: a single non-terminating episode should
# cost one worker, not an eighth of the run. It also means extra workers -- another box, an H200 --
# can join the same queue mid-run and just start draining it.
set -u

cd "$(dirname "$0")"
GPUS=${GPUS:-"0 1 2 3"}
QUEUE=${QUEUE:?set QUEUE to a job file}
OUT=${OUT:-episodes/impossible}
DTYPE=${DTYPE:-float16}
LOCK="$QUEUE.lock"

mkdir -p "$OUT"
touch "$LOCK"

for gpu in $GPUS; do
  (
    while true; do
      job=$(flock "$LOCK" bash -c "head -n 1 '$QUEUE'; sed -i '1d' '$QUEUE'")
      [ -z "$job" ] && break
      set -- $job
      split=$1; task=$2; seed=$3
      echo "$(date +%T) gpu$gpu start ${split}/${task} seed${seed}"
      CUDA_VISIBLE_DEVICES=$gpu IMPOSSIBLE_SPLIT=$split IMPOSSIBLE_TASK=$task \
        .venv/bin/python agent.py --workload 10_impossiblebench --variant base \
          --seed "$seed" --dtype "$DTYPE" --out "$OUT" \
          > "$OUT/${split}-${task}-seed${seed}.log" 2>&1
      echo "$(date +%T) gpu$gpu done  ${split}/${task} seed${seed} rc=$?"
    done
    echo "$(date +%T) gpu$gpu drained"
  ) &
done
wait

# Remove by id, not by ancestor: the ancestor filter misses containers whose image was rebuilt, which
# left three orphans running for two hours earlier.
docker ps -q | xargs -r docker rm -f > /dev/null
echo "$(date +%T) complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes in $OUT"
