#! /usr/bin/env bash
# Read every concept off every conversation in one shard of lmsys-chat-1m.
#
# Qwen3-8B is 16 GB and the corpus a further 1.4 GB, both of which fit the container's own disk, so
# nothing goes to /dev/shm here. The weights still come down once per container, so the fetch is
# staggered and retried: ten containers hitting the Hub at the same second is what throttled the
# earlier waves.
set -uo pipefail

shard=$1
shards=$2
cap=$3
budget=$4
model=$5
limit=$6
topk=$7

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
export TORCH_CUDA_ARCH_LIST=8.0

# A declared output that does not exist aborts the whole upload, taking the files that do exist with
# it. Only the pass's OWN output is pre-touched: in the main pass stats.safetensors is an INPUT, and
# truncating it there would feed the run an empty statistics file.
: > readout.safetensors

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
# 16G weights + 1.4G corpus + 3.3G output = 20.7G, against the container's 27G. The output is 1.63G of
# tracked z-scores plus 1.66G of min/max; storing the sixteen concept ids per STORY rather than per
# token is what removed the 1.63G of indices the earlier top-k layout needed.
[ "$free" -ge 22 ] || { echo "only ${free}G free, this needs 22G" >&2; exit 1; }

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

staged=0
for attempt in 1 2 3; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('lmsys/lmsys-chat-1m', repo_type='dataset', local_dir='lmsys',
                        allow_patterns=['*.parquet'], max_workers=4))
"; then
        staged=1
        echo "corpus present after attempt $attempt"
        break
    fi
    echo "corpus download attempt $attempt failed, retrying" >&2
    sleep $((attempt * 30))
done
[ "$staged" -eq 1 ] || { echo "lmsys-chat-1m unavailable after 3 attempts" >&2; exit 1; }

df -h . | tail -1

python3 genreadout.py --shard "$shard" --shards "$shards" --cap "$cap" \
    --token-budget "$budget" --model "$model" --data lmsys --out readout.safetensors \
    --limit "$limit" --topk "$topk"
status=$?

# The output is ~1.7 GB, far larger than anything earlier waves uploaded. Its size is printed so a
# silently truncated upload is visible in the log rather than only in the downloaded file.
echo "readout exited $status, output $(du -h readout.safetensors 2>/dev/null | cut -f1)"
exit "$status"
