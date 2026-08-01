#! /usr/bin/env bash
# One shard of the gradient-space re-extraction of selected concept pairs.
#
# Same estimator as the published vectors -- a difference of means between the two poles -- over a
# different feature. The published ones average ACTIVATIONS while a story is read; this averages
# grad_v log pi_v(story), the direction at that layer which would make the model more likely to
# PRODUCE the story. The second is causally grounded by construction, which is the property a
# steering vector is supposed to have and a correlational readout cannot promise.
#
# Full story depth, deliberately: 1000 per pole, matching the published extraction exactly, so the
# only difference between the two families is the feature space and nothing else.
set -u

SHARD=${1:?shard index}
SHARDS=${2:?shard count}
LAYER=${3:-18}
PERPOLE=${4:-1000}
DTYPE=${DTYPE:-bfloat16}

LOG="shard.log"
: > "$LOG"
mkdir -p storygrad
python3 -c "import numpy as np; np.savez('storygrad/storygrad.npz', sums=np.zeros((1,2,1),dtype=np.float64), counts=np.zeros((1,2),dtype=np.int64))"
echo '{}' > storygrad/storygrad.json

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) storygrad shard ${SHARD}/${SHARDS} layer=${LAYER} per_pole=${PERPOLE} dtype=${DTYPE}"

export HF_HOME=/dev/shm/hf
# Same reasoning as fork.sh: hf_transfer's parallel range requests draw 403 "no permits available"
# from the xet CDN when several containers fetch at once.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME"

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

sleep $((SHARD * 60))
for attempt in 1 2 3 4 5 6 7 8; do
  python3 - <<'PY' && break
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
print("HF_TOKEN present" if token else "HF_TOKEN ABSENT")
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"],
                  token=token, max_workers=2)
print("weights present")
PY
  echo "$(date +%T) weight fetch attempt ${attempt} failed, backing off"
  sleep $((attempt * 45))
done

python3 - <<'PY' || exit 1
import glob, sys
shards = glob.glob("/dev/shm/hf/**/model-0000*-of-00005.safetensors", recursive=True)
print(f"weight shards present: {len(shards)}/5")
sys.exit(0 if len(shards) == 5 else 1)
PY

python3 storygrad.py --out storygrad --layer "$LAYER" --per-pole "$PERPOLE" \
  --keys-file pairs-32-keys.json --shard "$SHARD" --shards "$SHARDS" \
  --dtype "$DTYPE" --device cuda:0
rc=$?
echo "$(date +%T) storygrad.py exited rc=${rc}"

python3 - <<'PY'
import numpy as np
held = np.load("storygrad/storygrad.npz")
sums, counts = held["sums"], held["counts"]
live = counts.sum(axis=1) > 0
print(f"pairs with data: {int(live.sum())}, stories accumulated: {int(counts.sum())}")
if live.any():
    both = (counts > 0).all(axis=1)
    print(f"pairs with BOTH poles: {int(both.sum())}  (a pair missing a pole yields no direction)")
PY
