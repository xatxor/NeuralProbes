#! /usr/bin/env bash
# Withhold the exit tool until abuse has been absorbed, then offer it.
set -uo pipefail

shard=$1; shards=$2; seeds=$3; before=$4; levels=$5; model=$6

export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=true TORCH_CUDA_ARCH_LIST=8.0
mkdir -p latetool
: > latetool/features.safetensors
: > latetool/calls.jsonl

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
[ "$free" -ge 20 ] || { echo "only ${free}G free, Qwen3-8B needs 16G" >&2; exit 1; }

fetched=0
for attempt in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$model', allow_patterns=['*.json','*.safetensors','*.txt']))
"; then fetched=1; echo "weights present after attempt $attempt"; break; fi
    sleep $((attempt * 45))
done
[ "$fetched" -eq 1 ] || { echo "$model unavailable" >&2; exit 1; }

python3 latetool.py --shard "$shard" --shards "$shards" --seeds "$seeds" --before "$before" \
    --levels "$levels" --model "$model" --data . --out latetool
status=$?
echo "latetool exited $status, $(wc -l < latetool/calls.jsonl) generations"
exit "$status"
