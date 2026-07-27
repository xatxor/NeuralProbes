#! /usr/bin/env bash
set -uo pipefail

model=$1
batch=$2

export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/dev/shm/hf
mkdir -p /dev/shm/hf

printf '{}' > translations.json

free=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9')
echo "tmpfs free: ${free}G"
[ "$free" -ge 56 ] || { echo "only ${free}G in /dev/shm; $model will not fit" >&2; exit 1; }

python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$model', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt']))
" || { echo "weights failed to download" >&2; exit 1; }
df -h /dev/shm | tail -1

# Inputs land at the path the spec declares, not flat beside the entrypoint.
python3 translate.py --pairs .bak/probes/pairs.parquet --out translations.json \
    --model "$model" --batch "$batch"
status=$?
# The exit status has to come from the work, not from whatever ran last. An `echo` here once let a
# failed translation report SUCCESS and upload a two-byte placeholder.
echo "translated $(wc -c < translations.json) bytes, translate.py exited $status"
exit "$status"
