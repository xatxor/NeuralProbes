#! /usr/bin/env bash
set -uo pipefail

shard=$1
shards=$2
rollouts=$3
tokens=$4
batch=$5

export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# A declared output that does not exist aborts the entire upload, taking the files that do exist with
# it. That is how the first run lost its generations as well as its judgements.
: > runs.jsonl
printf '{}' > manifest.json

free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
[ "$free" -ge 20 ] || { echo "only ${free}G free, Qwen3-8B needs 16G" >&2; exit 1; }

python3 screen.py --shard "$shard" --shards "$shards" --out . \
    --rollouts "$rollouts" --new-tokens "$tokens" --batch "$batch"
status=$?
echo "generation exited $status with $(wc -l < runs.jsonl) rows"
df -h . | tail -1
exit "$status"
