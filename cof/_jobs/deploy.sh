#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${DATASPHERE_PROJECT:?add DATASPHERE_PROJECT to .env or the environment}"
: "${HF_TOKEN:?export HF_TOKEN with access to Qwen/Qwen3-8B}"
: "${WORKERS:=10}"
command -v datasphere >/dev/null || { echo "install the datasphere CLI first" >&2; exit 127; }
command -v jq >/dev/null || { echo "install jq first" >&2; exit 127; }

if ((WORKERS < 1)); then
  echo "WORKERS must be at least 1" >&2
  exit 2
fi

shard_args() {
  local worker=$1
  shift
  printf '%q ' "$@" --num-workers "$WORKERS" --worker-index "$worker"
}

# User arguments come last, so they can override the Math-500 defaults.
base_args=(--benchmark math_500 --limit 30 "$@")
job=$(mktemp)
args=$(shard_args 0 "${base_args[@]}")
datasphere project job execute \
  -p "$DATASPHERE_PROJECT" \
  -c _jobs/steering.yaml \
  --async \
  -o "$job" \
  --env-var "HF_TOKEN=$HF_TOKEN" \
  --env-var "STEERING_ARGS=$args"
template=$(jq -r .job_id "$job")
echo "worker 0: $template"

while :; do
  datasphere project job get --id "$template" --format json -o "$job"
  status=$(jq -r .status "$job")
  case $status in
    EXECUTING | SUCCESS) break ;;
    CREATING | PREPARING) sleep 10 ;;
    *) echo "worker 0 is $status" >&2 && exit 1 ;;
  esac
done

failed=()
for ((worker = 1; worker < WORKERS; worker++)); do
  args=$(shard_args "$worker" "${base_args[@]}")
  if datasphere project job fork \
    --id "$template" \
    --async \
    -o "$job" \
    --env-var "STEERING_ARGS=$args"; then
    echo "worker $worker: $(jq -r .job_id "$job")"
  else
    failed+=("$worker")
  fi
done

if ((${#failed[@]})); then
  echo "${#failed[@]} workers failed to submit: ${failed[*]}" >&2
  exit 1
fi
