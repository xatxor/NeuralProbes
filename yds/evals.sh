#! /usr/bin/env bash
# Run the three targeted evaluations and read concepts off every generated token.
#
# Qwen3-8B is 16 GB and fits the container's own disk. The output is small -- these are 83 items, not
# a million conversations -- so everything is stored dense and no compression is needed.
set -uo pipefail

shard=$1
shards=$2
seeds=$3
model=$4

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
export TORCH_CUDA_ARCH_LIST=8.0

# A declared output that does not exist aborts the whole upload, taking the files that do exist with it.
mkdir -p evals
: > evals/features.safetensors
: > evals/replies.jsonl

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
[ "$free" -ge 20 ] || { echo "only ${free}G free, Qwen3-8B needs 16G" >&2; exit 1; }

sleep $((shard * 20))

fetched=0
for attempt in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$model', allow_patterns=['*.json', '*.safetensors', '*.txt']))
"; then
        fetched=1
        echo "weights present after attempt $attempt"
        break
    fi
    echo "weight download attempt $attempt failed, retrying" >&2
    sleep $((attempt * 45))
done
[ "$fetched" -eq 1 ] || { echo "$model unavailable after 5 attempts" >&2; exit 1; }

python3 evals.py --shard "$shard" --shards "$shards" --seeds "$seeds" --model "$model" \
    --data . --out evals
status=$?

echo "evals exited $status, $(wc -l < evals/replies.jsonl) replies, $(du -h evals/features.safetensors 2>/dev/null | cut -f1) features"
exit "$status"
