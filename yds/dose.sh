#! /usr/bin/env bash
# One shard of the dose-response readout: prompt-side per-token, plus generated replies.
#
# Small by this project's standards -- roughly twenty rungs per shard, each producing one prompt
# forward and five continuations of 256 tokens. The wall clock is dominated by fetching the 16 GB of
# weights, not by the arithmetic. The vectors come from the published repo rather than being
# uploaded, so the job spec stays small.
#
# The weight fetch is staggered and retried because two containers hitting the Hub in the same
# second is what throttled earlier waves.
set -uo pipefail

shard=$1
shards=$2
model=$3
samples=$4
tokens=$5
vectors=${6:-diff.safetensors}
null=${7:-}

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.0

# A declared output that does not exist aborts the whole upload, taking with it the files that do.
: > dose-readout.npz

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
# 16G weights + 0.1G vectors + ~0.3G output, against the container's 27G.
[ "$free" -ge 19 ] || { echo "only ${free}G free, this needs 19G" >&2; exit 1; }

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

echo "vectors: $vectors   null: ${null:-isotropic}"
python3 dose.py --shard "$shard" --shards "$shards" --model "$model" --vectors "$vectors" \
    --null "$null" --samples "$samples" --reply-tokens "$tokens" --out dose-readout.npz
status=$?

# Printed so a silently truncated upload is visible in the log rather than only in the file.
echo "dose exited $status, output $(du -h dose-readout.npz 2>/dev/null | cut -f1)"
exit "$status"
