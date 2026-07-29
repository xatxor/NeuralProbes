#! /usr/bin/env bash
# Label every lmsys-chat-1m prompt with the concept classes it could reveal.
#
# The disk layout is judgerun.sh's, which learned it the hard way: most of the container's 99 GB is
# the image, so the 54 GB Gemma weights go in /dev/shm, a 59 GB tmpfs. The dataset is only 1.2 GB
# and goes on the regular disk, where it does not compete with the weights for tmpfs.
#
# The weight download is staggered and retried for the same reason as in wave B: seven containers
# pulling 54 GB each at once throttles the Hub badly enough to kill shards outright.
set -uo pipefail

shard=$1
shards=$2
top=$3
maxchars=$4
model=$5

export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/dev/shm/hf
mkdir -p /dev/shm/hf

# A declared output that does not exist aborts the whole upload, taking the files that do exist with it.
: > labels.jsonl

# Printed first so shard 0 reveals immediately whether the vLLM image's entrypoint was replaced and
# whether its vllm is still importable after the requirements install.
echo "python: $(command -v python3)"
echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

free=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9')
echo "tmpfs free: ${free}G"
[ "$free" -ge 56 ] || { echo "only ${free}G in /dev/shm; $model will not fit" >&2; exit 1; }

sleep $((shard * 45))

fetched=0
for attempt in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$model', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
                        max_workers=4))
"; then
        fetched=1
        echo "weights present after attempt $attempt"
        break
    fi
    echo "weight download attempt $attempt failed, retrying" >&2
    sleep $((attempt * 60))
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

df -h /dev/shm . | tail -3

python3 label.py --shard "$shard" --shards "$shards" --top "$top" --max-chars "$maxchars" \
    --model "$model" --data lmsys --classes classes.json --out labels.jsonl
status=$?
echo "labelling exited $status with $(wc -l < labels.jsonl) rows"
exit "$status"
