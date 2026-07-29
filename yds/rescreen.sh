#! /usr/bin/env bash
# Generate the re-screen: every concept vector steered on prompts from its own class.
#
# Qwen3-8B is 16 GB and fits the container's own disk, unlike the 54 GB judge that forced /dev/shm
# in wave B. The weights still come down once per container, so the fetch is staggered and retried:
# ten containers hitting the Hub at the same second is what throttled the earlier waves.
set -uo pipefail

shard=$1
shards=$2
layer=$3
alpha=$4
tokens=$5
chunk=$6
baseline=$7

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
export TORCH_CUDA_ARCH_LIST=8.0

# A declared output that does not exist aborts the whole upload, taking the files that do exist with it.
: > runs.jsonl
printf '{}' > manifest.json

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
[ "$free" -ge 20 ] || { echo "only ${free}G free, Qwen3-8B needs 16G" >&2; exit 1; }

sleep $((shard * 20))

fetched=0
for attempt in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B', allow_patterns=['*.json', '*.safetensors', '*.txt']))
"; then
        fetched=1
        echo "weights present after attempt $attempt"
        break
    fi
    echo "weight download attempt $attempt failed, retrying" >&2
    sleep $((attempt * 45))
done
[ "$fetched" -eq 1 ] || { echo "Qwen3-8B unavailable after 5 attempts" >&2; exit 1; }

# Both layers run in one container. Weights cost ten minutes to fetch and the model a further
# minute to load, so paying that twice to steer at two depths wastes an hour across the fleet. The
# layer is recorded on every row, so the passes append into one file and the judge separates them.
#
# The unsteered baseline does not depend on the layer, so only the first pass emits it. Passing
# --baseline to every shard is correct: rescreen.py splits those chunks across shards, so no row is
# generated twice.
status=0
for pass in $(echo "$layer" | tr ',' ' '); do
    extra=""
    [ "$baseline" = "1" ] && [ ! -s runs-done.jsonl ] && extra="--baseline"

    echo "=== layer L${pass} ==="
    python3 rescreen.py --out . --shard "$shard" --shards "$shards" --layer "$pass" \
        --alpha "$alpha" --new-tokens "$tokens" --chunk "$chunk" $extra
    status=$?
    [ "$status" -eq 0 ] || { echo "L${pass} exited $status, stopping" >&2; break; }

    cat runs.jsonl >> runs-done.jsonl
    cp manifest.json "manifest-L${pass}.json"
    echo "L${pass} done, $(wc -l < runs.jsonl) rows, $(wc -l < runs-done.jsonl) cumulative"
done

# The declared output is the union of the passes; the per-layer manifests are folded in beside it.
mv runs-done.jsonl runs.jsonl 2>/dev/null || true
python3 -c "
import glob, json
json.dump({f.split('.')[0].split('-')[-1]: json.load(open(f)) for f in sorted(glob.glob('manifest-L*.json'))},
          open('manifest.json','w'), indent=2)
" 2>/dev/null || true

echo "generation exited $status with $(wc -l < runs.jsonl) rows"
df -h . | tail -1
exit "$status"
