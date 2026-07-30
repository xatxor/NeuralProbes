#! /usr/bin/env bash
# Pull every readout shard onto the long-lived server, not onto the Mac.
#
# Each shard is ~1.7 GB. The Mac's datasphere CLI still carries the stock 1 GB cap, over which it
# logs a note, SKIPS the file, and exits 0 -- a silent loss that looks exactly like success. The
# server's copy is patched to 64 GB (datasphere/files.py:107, original kept as files.py.orig), so
# collection happens there.
#
# The CLI authenticates from YC_IAM_TOKEN. The name matters: IAM_TOKEN is a substring of it and is
# silently ignored (auth.py:33). Tokens last about 12 hours and are minted on the Mac, because the
# server has no yc CLI and .env must never land on a shared box.
#
# Needs SSH_PASS_FILE pointing at a file holding the server password. The SSH_PASS in .env is stale
# and opens only User22; it does not work for this box.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SSH_PASS_FILE:?point it at a file holding the User18 password}"

host=${HOST:-User18@176.109.111.31}
ledger=${LEDGER:-.bak/readout-shards.log}
target=${TARGET:-neuralprobes/readout}
[ -f "$ledger" ] || { echo "no ledger at $ledger" >&2; exit 1; }

token=$(yc iam create-token) || { echo "could not mint an IAM token" >&2; exit 1; }

remote() {
  sshpass -f "$SSH_PASS_FILE" ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no \
    -o PreferredAuthentications=keyboard-interactive -o ConnectTimeout=20 "$host" "$@"
}

# The last id recorded for a shard wins: a re-forked shard appears twice, and the earlier line is the
# job that died. A monitor that takes the first would poll a corpse.
declare -A latest
while read -r shard id; do
  case "$id" in "" | FORK | null) continue ;; esac
  latest[$shard]=$id
done < <(sed -n 's/^shard \([0-9]*\): \([a-z0-9]*\)$/\1 \2/p' "$ledger")

echo "collecting ${#latest[@]} shards onto $host:$target"
remote "mkdir -p $target"

failed=()
for shard in $(printf '%s\n' "${!latest[@]}" | sort -n); do
  id=${latest[$shard]}
  dir=$(printf '%s/shard-%03d' "$target" "$shard")
  echo "--- shard $shard ($id) ---"
  if remote "YC_IAM_TOKEN='$token' mkdir -p $dir && cd $dir && YC_IAM_TOKEN='$token' \
      ~/dsvenv/bin/datasphere project job download-files --id $id --output-dir ." >/dev/null 2>&1; then
    size=$(remote "du -sh $dir 2>/dev/null | cut -f1")
    echo "shard $shard: $size"
  else
    echo "shard $shard: DOWNLOAD FAILED"
    failed+=("$shard")
  fi
done

echo "=== on $host:$target ==="
remote "du -sh $target; find $target -name readout.safetensors -printf '%p %s\n' | sort"

if ((${#failed[@]})); then
  echo "${#failed[@]} shards failed to download: ${failed[*]}" >&2
  exit 1
fi
