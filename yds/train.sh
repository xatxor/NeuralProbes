#! /usr/bin/env bash
# Train a steering vector against the TRUE DPO objective on prefix-matched branch points.
#
# Not the linearisation. Lemma 1 bounds the first-order surrogate's error by O(||v||^2), so it
# describes the real loss only for small v -- and a vector that steers is not small. This optimises v
# with forward and backward passes through the frozen model, inside a norm ball chosen as the trust
# region rather than stumbled into.
set -u

LAYER=${1:-18}
RADIUS=${2:-50}
WINDOW=${3:-thinking}
EPOCHS=${4:-6}
LR=${5:-0.05}
RADIUS_NOTE="the norm ball IS the regulariser here: held-out fell as |v| grew past ~11"
DTYPE=${DTYPE:-bfloat16}

LOG="job.log"
: > "$LOG"
mkdir -p vectors
python3 -c "import numpy as np; np.save('vectors/placeholder.npy', np.zeros(4, dtype=np.float32))"

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) train layer=${LAYER} radius=${RADIUS} window=${WINDOW} epochs=${EPOCHS}"

export HF_HOME=/dev/shm/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
# Fragmentation was 7.9 GB of the 80 GB card on the first attempt.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
tar -xzf forked-all.tar.gz
echo "$(date +%T) $(ls forked/*.json 2>/dev/null | wc -l) continuations"

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

python3 bipo.py train --dir forked --out vectors --layer "$LAYER" --radius "$RADIUS" \
  --window "$WINDOW" --epochs "$EPOCHS" --lr "$LR" --dtype "$DTYPE" --device cuda:0 --max-tokens 9000
rc=$?
echo "$(date +%T) train exited rc=${rc}"
rm -f vectors/placeholder.npy
ls -la vectors/
