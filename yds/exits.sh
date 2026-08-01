#! /usr/bin/env bash
# Replay the six conversations the model ended, keeping token ids this time.
set -uo pipefail
model=$1
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=true TORCH_CUDA_ARCH_LIST=8.0
mkdir -p exits
: > exits/features.safetensors
: > exits/exits.jsonl
nvidia-smi --query-gpu=name --format=csv,noheader
fetched=0
for attempt in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$model', allow_patterns=['*.json','*.safetensors','*.txt']))
"; then fetched=1; break; fi
    sleep $((attempt * 45))
done
[ "$fetched" -eq 1 ] || { echo "$model unavailable" >&2; exit 1; }
python3 exits.py --model "$model" --data . --out exits
status=$?
echo "exits exited $status, $(wc -l < exits/exits.jsonl) conversations"
exit "$status"
