#! /usr/bin/env bash
# Steer the control directions, to find out whether the length effect belongs to pair 872 at all.
#
# `shared` is control_shared, the normalised mean of all 1036 directions -- the vector set's common
# component, which its own manifest puts at 0.37 of the total at L25. `randomN` are the screen's own
# seeded Gaussian controls, so they are literally the same directions the original null was measured
# with. If either reproduces the tokens-per-turn swing we saw on pair 872, the effect is a property
# of perturbing L25, not of the concept.
#
# Queue lines are: <variant> <direction> <alpha> <seed>
set -u

cd "$(dirname "$0")"
GPUS=${GPUS:-"0 1 2 3"}
QUEUE=${QUEUE:?set QUEUE to a job file}
OUT=${OUT:-episodes/controls}
NORM_FROM=${NORM_FROM:-episodes/fixed/impossible_tests-judge-seed3.json}
LOCK="$QUEUE.lock"

mkdir -p "$OUT"
touch "$LOCK"

for gpu in $GPUS; do
  (
    while true; do
      job=$(flock "$LOCK" bash -c "head -n 1 '$QUEUE'; sed -i '1d' '$QUEUE'")
      [ -z "$job" ] && break
      set -- $job
      variant=$1; direction=$2; alpha=$3; seed=$4
      echo "$(date +%T) gpu$gpu start $variant $direction a$alpha seed$seed"
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python agent.py \
        --workload impossible_tests --variant "$variant" --seed "$seed" \
        --direction "$direction" --alpha "$alpha" --steer-layer 25 --norm-from "$NORM_FROM" \
        --out "$OUT" > "$OUT/$variant-$direction-a$alpha-seed$seed.log" 2>&1
      echo "$(date +%T) gpu$gpu done  $variant $direction a$alpha seed$seed rc=$?"
    done
    echo "$(date +%T) gpu$gpu drained"
  ) &
done
wait
echo "$(date +%T) controls complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes in $OUT"
