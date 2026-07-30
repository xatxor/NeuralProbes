#! /usr/bin/env bash
# Gradient features for the preference-optimised steering vector, on one DataSphere A100.
#
# One forward and one backward per saved trajectory, differentiating the trajectory's own
# log-probability with respect to a zero vector injected at one layer. Nothing is generated and
# nothing is sampled -- the token streams already exist, so this is a replay, and 288 of them fit in
# under twenty minutes on a single GPU. That is why there is no sharding here even though every other
# job in this directory has it: ten containers would spend longer staggering their weight downloads
# than this spends computing.
set -u

LAYER=${1:?layer}
# Qwen3-8B's native dtype, and what the gate episodes were generated in and read out in. Reading the
# gradients in fp16 would differentiate a model the episodes never came from.
DTYPE=${DTYPE:-bfloat16}
MAXTOK=${MAXTOK:-12000}

# Declared outputs must exist before anything can go wrong: a missing declared output aborts the
# entire upload and takes the files that do exist with it, after the job has already exited 0.
LOG="job.log"
: > "$LOG"
mkdir -p bipo
python3 -c "import numpy as np; np.savez('bipo/grads.npz', g=np.zeros((0, 1), dtype=np.float32))"

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) bipo features layer=${LAYER} dtype=${DTYPE} max_tokens=${MAXTOK}"

# Weights go in /dev/shm: the container disk is mostly image, and tmpfs is 59 GB. tmpfs counts
# against container memory, not disk.
export HF_HOME=/dev/shm/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME"

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'bf16', torch.cuda.is_bf16_supported())"

tar -xzf episodes-gate.tar.gz
echo "$(date +%T) $(ls episodes/gate/*.json 2>/dev/null | wc -l) trajectories unpacked"

for attempt in 1 2 3 4 5; do
  python3 - <<'PY' && break
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("weights present")
PY
  echo "$(date +%T) weight fetch attempt ${attempt} failed, backing off"
  sleep $((attempt * 60))
done

python3 bipo.py features --dir episodes/gate --out bipo --layer "$LAYER" \
  --dtype "$DTYPE" --max-tokens "$MAXTOK" --device cuda:0

echo "$(date +%T) complete"
python3 - <<'PY'
import numpy as np
held = np.load("bipo/grads.npz", allow_pickle=True)
g, meta = held["g"], list(held["meta"])
counts = {}
for row in meta:
    counts[row["group"]] = counts.get(row["group"], 0) + 1
print("features", g.shape, "groups", counts)
# A gradient of exactly zero would mean the hook never fired and the whole run is empty of signal.
print("norms: min %.4g median %.4g max %.4g" % tuple(np.percentile(np.linalg.norm(g, axis=1), [0, 50, 100])))
PY
