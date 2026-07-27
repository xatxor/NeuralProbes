#! /usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a

log=${LOG:-.bak/gen-deploy.log}
job=$(mktemp)

mapfile -t ids < <(sed -n 's/^shard \([0-9]*\): \(.*\)$/\2/p' "$log")
if ((${#ids[@]} == 0)); then
    echo "no shards launched yet in $log"
    tail -3 "$log"
    exit 0
fi

printf '%-6s %-22s %s\n' shard job status
for index in "${!ids[@]}"; do
    state=UNREACHABLE
    for _ in 1 2 3; do
        if datasphere project job get --id "${ids[index]}" --format json -o "$job" >/dev/null 2>&1; then
            value=$(jq -r .status "$job" 2>/dev/null)
            [ -n "$value" ] && [ "$value" != "null" ] && state=$value && break
        fi
        sleep 3
    done
    printf '%-6s %-22s %s\n' "$index" "${ids[index]}" "$state"
done
