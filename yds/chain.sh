#! /usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
source .env
set +a

job=$(mktemp)
smoke_config=${SMOKE:?a small spec to run first}
datasphere project job execute -p "$DATASPHERE_PROJECT" -c "$smoke_config" --async -o "$job"
smoke=$(jq -r .job_id "$job")
echo "smoke: $smoke"

while :; do
  datasphere project job get --id "$smoke" --format json -o "$job" >/dev/null
  state=$(jq -r .status "$job")
  case "$state" in
    SUCCESS) break ;;
    CREATING | PREPARING | EXECUTING | UPLOADING_OUTPUT) sleep 15 ;;
    *) echo "smoke FAILED with $state" >&2 && exit 1 ;;
  esac
done

rm -rf '$blob-smoke'
mkdir -p '$blob-smoke'
datasphere project job download-files --id "$smoke" --output-dir '$blob-smoke'
echo "smoke OK, files:"
ls -la '$blob-smoke'

echo "=== launching full grid ==="
exec yds/deploy.sh
