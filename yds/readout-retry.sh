#! /usr/bin/env bash
# Keep trying to place the readout smoke until DataSphere has a free A100.
#
# When the pool is full a job ERRORs within seconds of creation, with no status_details at all -- the
# same signature as the VM allocation failures in earlier waves. That is indistinguishable from a
# real crash by status alone, so the two are told apart by HOW LONG the job lived: an allocation
# failure dies in seconds, whereas a job that got a VM and then broke has spent minutes pulling 16 GB
# of weights first.
#
# A fast ERROR means retry. A slow ERROR means the code is wrong and retrying would just burn
# capacity, so the loop stops and says so.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
source .env
set +a
: "${DATASPHERE_PROJECT:?add it to .env}"

config=${CONFIG:-yds/readout-smoke.yaml}
interval=${INTERVAL:-1800}
attempts=${ATTEMPTS:-48}
# Below this many seconds alive, an ERROR is read as "no capacity" rather than "the job is broken".
threshold=${THRESHOLD:-120}
ledger=${LEDGER:-.bak/readout-retry.log}
mkdir -p "$(dirname "$ledger")"

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$ledger"; }

job=$(mktemp)
status() {
  local value=""
  for _ in 1 2 3 4 5; do
    if datasphere project job get --id "$1" --format json -o "$job" >/dev/null 2>&1; then
      value=$(jq -r .status "$job" 2>/dev/null)
      [ -n "$value" ] && [ "$value" != "null" ] && break
    fi
    sleep 5
  done
  echo "${value:-UNREACHABLE}"
}

say "starting: $config every ${interval}s, up to $attempts attempts"

for ((attempt = 1; attempt <= attempts; attempt++)); do
  started=$(date +%s)
  if ! datasphere project job execute -p "$DATASPHERE_PROJECT" -c "$config" --async -o "$job" >/dev/null 2>&1; then
    say "attempt $attempt: submit failed outright"
    sleep "$interval"
    continue
  fi
  id=$(jq -r .job_id "$job" 2>/dev/null)
  if [ -z "$id" ] || [ "$id" = "null" ]; then
    say "attempt $attempt: no job id returned"
    sleep "$interval"
    continue
  fi
  say "attempt $attempt: $id"

  # Watch this attempt closely. Anything that reaches EXECUTING has a VM and the wait is over.
  verdict=""
  while :; do
    state=$(status "$id")
    case "$state" in
      EXECUTING | UPLOADING_OUTPUT | SUCCESS)
        verdict="up"
        break
        ;;
      CREATING | PREPARING)
        sleep 15
        ;;
      *)
        verdict="$state"
        break
        ;;
    esac
  done
  alive=$(( $(date +%s) - started ))

  if [ "$verdict" = "up" ]; then
    say "GOT A GPU on attempt $attempt after ${alive}s: $id is $(status "$id")"
    say "smoke is running; the ten-shard deploy is NOT started automatically"
    exit 0
  fi

  if [ "$alive" -ge "$threshold" ]; then
    say "attempt $attempt: $verdict after ${alive}s -- too slow to be a capacity failure, STOPPING"
    say "inspect with: datasphere project job attach --id $id"
    exit 1
  fi

  say "attempt $attempt: $verdict after ${alive}s (no capacity), waiting ${interval}s"
  sleep "$interval"
done

say "gave up after $attempts attempts"
exit 1
