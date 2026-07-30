#! /usr/bin/env bash
# Deploy the concept readout across ten A100 containers.
#
# Unlike deploy.sh this never downloads. Each shard writes ~1.7 GB, and the Mac's datasphere CLI
# still has the stock 1 GB cap, over which it logs a note, SKIPS THE FILE, and exits 0. The output
# is collected on the server instead, whose CLI is patched to 64 GB.
#
# Forks go out all at once first, because when DataSphere is healthy that is simply faster. Any that
# come back ERROR are re-forked one at a time, waiting for each to reach EXECUTING before the next --
# which is the pattern that works when it is not healthy.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
source .env
set +a
: "${DATASPHERE_PROJECT:?add it to .env}"
: "${HF_REPO:?add it to .env}"

config=${CONFIG:-yds/readout.yaml}
ledger=${LEDGER:-.bak/$(basename "$config" .yaml)-shards.log}
mkdir -p "$(dirname "$ledger")"
printf '# %s %s\n' "$(date -u +%FT%TZ)" "$config" >> "$ledger"

shards=$(sed -n 's/^[[:space:]]*SHARDS: "\([0-9]*\)"/\1/p' "$config")
: "${shards:?could not read SHARDS from $config}"
job=$(mktemp)
declare -a ids

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

# The ledger entry is written BEFORE anything else is done with a job id. A fork whose id is not
# recorded is invisible, and a monitor then polls a dead id forever while the real job runs on.
launch() {
  local shard=$1 id=""
  if [ "$shard" -eq 0 ]; then
    datasphere project job execute -p "$DATASPHERE_PROJECT" -c "$config" --async -o "$job" >/dev/null 2>&1
  else
    datasphere project job fork --id "${ids[0]}" --async --arg SHARD="$shard" --arg SHARDS="$shards" -o "$job" >/dev/null 2>&1
  fi
  id=$(jq -r .job_id "$job" 2>/dev/null)
  # An empty id means the fork itself failed. Recording it would poison the ledger with a job that
  # can never be polled, so it is reported instead.
  if [ -z "$id" ] || [ "$id" = "null" ]; then
    echo "shard $shard: FORK FAILED" | tee -a "$ledger"
    return 1
  fi
  echo "shard $shard: $id" | tee -a "$ledger"
  ids[shard]=$id
}

launch 0 || exit 1
while :; do
  case "$(status "${ids[0]}")" in
    EXECUTING | UPLOADING_OUTPUT | SUCCESS) break ;;
    CREATING | PREPARING) sleep 10 ;;
    *) echo "shard 0 never started" >&2 && exit 1 ;;
  esac
done
echo "shard 0 is up; forking the rest all at once"

for ((shard = 1; shard < shards; shard++)); do
  launch "$shard" || true
done

# Give the wave a moment to declare itself, then re-fork anything that fell over. A shard that ERRORs
# within ~40s with empty outputs is a VM allocation failure, and re-forking is the standing fix.
sleep 60
for ((shard = 1; shard < shards; shard++)); do
  state=$(status "${ids[shard]:-none}")
  case "$state" in
    CREATING | PREPARING | EXECUTING | UPLOADING_OUTPUT | SUCCESS) continue ;;
  esac
  echo "shard $shard came back $state, re-forking one at a time"
  for attempt in 1 2 3; do
    launch "$shard" || { sleep 30; continue; }
    while :; do
      case "$(status "${ids[shard]}")" in
        EXECUTING | UPLOADING_OUTPUT | SUCCESS) break 2 ;;
        CREATING | PREPARING) sleep 10 ;;
        *) echo "shard $shard attempt $attempt failed" >&2 && sleep 30 && break ;;
      esac
    done
  done
done

echo "=== all shards registered ==="
for ((shard = 0; shard < shards; shard++)); do
  printf 'shard %d: %s %s\n' "$shard" "${ids[shard]:-MISSING}" "$(status "${ids[shard]:-none}")"
done
