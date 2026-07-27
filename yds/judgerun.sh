#! /usr/bin/env bash
# Wave B: blinded pairwise judging. The judge is the only model this container downloads.
#
# The weights go in /dev/shm, a 59 GB tmpfs: the container's disk is 99 GB with 68 GB taken by the
# image, so 28 GB is all that is free and a 54 GB model cannot live there.
#
# The download is retried. Ten containers pulling 54 GB each is 540 GB from Hugging Face at once, which
# throttles the transfer from five minutes to ten and, on the first attempt at this, killed seven of ten
# shards outright twenty minutes in. `snapshot_download` resumes from what it already has, so a retry is
# cheap and only fetches the remainder.
set -uo pipefail

mode=${5:-pairwise}
# Inputs land at the path the spec declares, not flat beside the entrypoint.
indir=${6:-screen}
shard=$1
shards=$2
jbatch=$3
judge=$4

export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/dev/shm/hf
mkdir -p /dev/shm/hf

# A declared output that does not exist aborts the whole upload, taking the files that do exist with it.
: > labels.jsonl

free=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9')
echo "tmpfs free: ${free}G"
[ "$free" -ge 56 ] || { echo "only ${free}G in /dev/shm; $judge will not fit" >&2; exit 1; }

# Stagger by shard so ten containers do not hit the Hub in the same second.
sleep $((shard * 45))

fetched=0
for attempt in 1 2 3 4 5; do
    if python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$judge', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
                        max_workers=4))
"; then
        fetched=1
        echo "weights present after attempt $attempt"
        break
    fi
    echo "download attempt $attempt failed, retrying" >&2
    sleep $((attempt * 60))
done
[ "$fetched" -eq 1 ] || { echo "judge weights unavailable after 5 attempts" >&2; exit 1; }
df -h /dev/shm | tail -1

python3 judge.py --input "$indir" --out labels.jsonl --mode "$mode" --model "$judge" \
    --shard "$shard" --shards "$shards" --batch "$jbatch"
