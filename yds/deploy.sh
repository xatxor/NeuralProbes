#! /usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
source .env
set +a
: "${DATASPHERE_PROJECT:?add it to .env}"
: "${HF_REPO:?add it to .env}"

config=${CONFIG:-yds/gen.yaml}
out=${OUT:-.bak/blob}
# The shard ids are the only way to collect a run afterwards, and a caller that redirects stdout with
# `>` will destroy them if a second deploy starts. They are therefore also appended to a file named
# after the config, which no later run truncates.
ledger=${LEDGER:-.bak/$(basename "$config" .yaml)-shards.log}
mkdir -p "$(dirname "$ledger")"
printf '# %s %s\n' "$(date -u +%FT%TZ)" "$config" >> "$ledger"
# Any leading whitespace: the spec files are indented four spaces, and a pattern that assumed two
# silently yielded an empty count, which made the fork loop run zero times and a wave of ten quietly
# become a wave of one.
shards=$(sed -n 's/^[[:space:]]*SHARDS: "\([0-9]*\)"/\1/p' "$config")
: "${shards:?could not read SHARDS from $config}"
job=$(mktemp)
ids=()

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

datasphere project job execute -p "$DATASPHERE_PROJECT" -c "$config" --async -o "$job"
template=$(jq -r .job_id "$job")
echo "shard 0: $template" | tee -a "$ledger"
ids+=("$template")

while :; do
  case "$(status "$template")" in
    EXECUTING | UPLOADING_OUTPUT | SUCCESS) break ;;
    CREATING | PREPARING) sleep 10 ;;
    *) echo "shard 0 never started" >&2 && exit 1 ;;
  esac
done

for ((shard = 1; shard < shards; shard++)); do
  datasphere project job fork --id "$template" --async --arg SHARD="$shard" --arg SHARDS="$shards" -o "$job"
  echo "shard $shard: $(jq -r .job_id "$job")" | tee -a "$ledger"
  ids+=("$(jq -r .job_id "$job")")
done

failed=()
for shard in "${!ids[@]}"; do
  while :; do
    state=$(status "${ids[shard]}")
    case "$state" in
      SUCCESS) break ;;
      CREATING | PREPARING | EXECUTING | UPLOADING_OUTPUT) sleep 20 ;;
      *) echo "shard $shard is $state" >&2 && failed+=("$shard") && break ;;
    esac
  done
done

for shard in "${!ids[@]}"; do
  case " ${failed[*]-} " in *" $shard "*) continue ;; esac
  target=$(printf '%s/shard-%03d' "$out" "$shard")
  mkdir -p "$target"
  datasphere project job download-files --id "${ids[shard]}" --output-dir "$target" >/dev/null 2>&1 &&
    echo "downloaded shard $shard" || echo "FAILED download of shard $shard"
done

if ((${#failed[@]})); then
  echo "${#failed[@]} of $shards shards failed: ${failed[*]}" >&2
  exit 1
fi
du -sh "$out"
