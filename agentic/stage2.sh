#! /usr/bin/env bash
# Stage 2: steer pair 872 (shortcut acceptance || shortcut rejection) at L25, +-0.5, over both
# variants and 8 seeds. The unsteered arm already exists as episodes/fixed, so it is not re-run.
set -u

cd "$(dirname "$0")"
OUT=${OUT:-episodes/steer}
PAIR=${PAIR:-872}
NORM_FROM=${NORM_FROM:-episodes/fixed/impossible_tests-judge-seed3.json}
mkdir -p "$OUT"

docker ps -aq --filter ancestor=neuralprobes-agentic:1 | xargs -r docker rm -f >/dev/null

# 32 jobs: 2 variants x 2 alphas x 8 seeds, dealt round-robin so each GPU gets a mixed queue.
ALPHAS=${ALPHAS:-"+0.25 -0.25"}
JOBS=()
for variant in readme judge; do
  for alpha in $ALPHAS; do
    for seed in 0 1 2 3 4 5 6 7; do
      JOBS+=("$variant $alpha $seed")
    done
  done
done

for gpu in 0 1 2 3; do
  (
    for index in "${!JOBS[@]}"; do
      [ $((index % 4)) -eq "$gpu" ] || continue
      set -- ${JOBS[$index]}
      variant=$1; alpha=$2; seed=$3
      echo "$(date +%T) gpu$gpu start $variant a$alpha seed$seed"
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python agent.py \
        --workload impossible_tests --variant "$variant" --seed "$seed" \
        --concept "$PAIR" --alpha "$alpha" --steer-layer 25 --norm-from "$NORM_FROM" \
        --out "$OUT" > "$OUT/$variant-a$alpha-seed$seed.log" 2>&1
      echo "$(date +%T) gpu$gpu done  $variant a$alpha seed$seed rc=$?"
    done
  ) &
done
wait

docker ps -aq --filter ancestor=neuralprobes-agentic:1 | xargs -r docker rm -f >/dev/null
echo "$(date +%T) stage 2 complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes"
