#! /usr/bin/env bash
# Put everything the judging box needs onto it, and verify it arrived.
#
# The box is rented with the cards already attached, so this runs once and then judgebox-setup.sh
# does the 63 GB model pull. Only four things have to travel: the judge, the two lookup tables it
# resolves pair ids and prompt slots through, and the generations themselves.
#
# `pairs.parquet` comes from `.bak/probes/`, the vectors the generations were actually produced with.
# The re-extracted set in `probes-notemplate/` has an identical pair order -- that was checked, not
# assumed -- but the file that matches the data is the one to ship.
#
# Usage:  HOST=ubuntu@1.2.3.4 KEY=~/Downloads/box.pem ./judgebox-upload.sh
set -uo pipefail

: "${HOST:?set HOST, e.g. ubuntu@1.2.3.4}"
KEY=${KEY:-}
root=${ROOT:-judge}

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ssh_opts=(-o StrictHostKeyChecking=no -o ConnectTimeout=25)
[ -n "$KEY" ] && ssh_opts+=(-i "$KEY")
SSH=(ssh "${ssh_opts[@]}")

echo "=== creating layout on $HOST ==="
"${SSH[@]}" "$HOST" "mkdir -p ~/$root/runs" || { echo "cannot reach $HOST" >&2; exit 1; }

echo "=== scripts and lookup tables ==="
rsync -az --stats -e "${SSH[*]}" \
    judge.py .bak/probes/pairs.parquet prompts.json \
    yds/judgebox-setup.sh yds/judgebox-smoke.sh yds/judgebox-run.sh \
    "$HOST:~/$root/" || { echo "rsync of scripts FAILED" >&2; exit 1; }

echo "=== generations (223 MB, 10 shards) ==="
rsync -az --stats -e "${SSH[*]}" .bak/rescreen/ "$HOST:~/$root/runs/" \
    || { echo "rsync of generations FAILED" >&2; exit 1; }

echo "=== verifying what landed ==="
local_rows=$(cat .bak/rescreen/shard-*/runs.jsonl | wc -l | tr -d ' ')
"${SSH[@]}" "$HOST" "cd ~/$root && \
    echo \"runs.jsonl files : \$(find runs -name runs.jsonl | wc -l)\" && \
    echo \"generation rows  : \$(cat runs/shard-*/runs.jsonl | wc -l)\" && \
    echo \"judge.py         : \$(wc -l < judge.py) lines\" && \
    echo \"pairs.parquet    : \$(stat -c%s pairs.parquet) bytes\" && \
    echo \"prompts.json     : \$(stat -c%s prompts.json) bytes\" && \
    chmod +x judgebox-*.sh"
echo "local generation rows: $local_rows  (both numbers must match)"

cat <<EOF

=== next, on the box ===
  ./judgebox-setup.sh                                          # ~63 GB pull, pins vllm==0.26.0
  COMPARISONS=plus_baseline,minus_baseline ./judgebox-smoke.sh 500
  COMPARISONS=plus_baseline,minus_baseline ./judgebox-run.sh 2 medium 2048

Read the smoke output before starting the run. It reports verdicts/s and the projected wall clock;
the first sweep measured 16.8 verdicts/s per H200 and finished 84,288 verdicts in 83.5 minutes.
EOF
