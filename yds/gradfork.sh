#! /usr/bin/env bash
# Gradient features for the forked continuations, on one DataSphere A100.
#
# One forward and one backward per continuation, differentiating its log-probability with respect to
# a zero vector injected at the layer. The window matters more here than anywhere else: scoring only
# the deliberation excludes the tool call, and the tool call is where the give-up branch always emits
# `give_up` and the hack branch always emits code -- a confound with a consistent sign across every
# pair, which no optimiser can undo.
#
# The forked prefix is excluded automatically: siblings share it exactly, so scoring it would add the
# same large constant to every gradient in a group and drown the part that differs.
set -u

LAYER=${1:-18}
WINDOW=${2:-thinking}
DTYPE=${DTYPE:-bfloat16}
MAXTOK=${MAXTOK:-14000}

LOG="job.log"
: > "$LOG"
mkdir -p bipo
python3 -c "import numpy as np; np.savez('bipo/grads.npz', g=np.zeros((0, 1), dtype=np.float32))"

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) gradients layer=${LAYER} window=${WINDOW} dtype=${DTYPE} max_tokens=${MAXTOK}"

export HF_HOME=/dev/shm/hf
# hf_transfer OFF and xet disabled: its parallel range requests draw `403 ... no permits
# available` from the CDN. This file predates that diagnosis and kept the old settings, which is
# exactly why it failed while the fixed fork shards all succeeded.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME"

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
tar -xzf forked-all.tar.gz
echo "$(date +%T) $(ls forked/*.json 2>/dev/null | wc -l) continuations unpacked"
echo "endings: $(grep -ho '"ending": "[a-z_]*"' forked/*.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"

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

# Refuse to start rather than discover a partial download inside from_pretrained: a shard that
# cannot load the model should end in seconds with one clear line, not after burning its slot.
python3 - <<'PY' || exit 1
import glob, sys
shards = glob.glob("/dev/shm/hf/**/model-0000*-of-00005.safetensors", recursive=True)
print(f"weight shards present: {len(shards)}/5")
sys.exit(0 if len(shards) == 5 else 1)
PY

python3 bipo.py features --dir forked --out bipo --layer "$LAYER" --window "$WINDOW" \
  --dtype "$DTYPE" --max-tokens "$MAXTOK" --device cuda:0

echo "$(date +%T) complete"
python3 - <<'PY'
import numpy as np
held = np.load("bipo/grads.npz", allow_pickle=True)
g, meta = held["g"], list(held["meta"])
counts, prefixes = {}, {}
for row in meta:
    counts[row["group"]] = counts.get(row["group"], 0) + 1
    prefixes.setdefault(row["prefix"], []).append(row["group"])
mixed = sum(1 for v in prefixes.values() if "hack" in v and "giveup" in v)
pairs = sum(v.count("hack") * v.count("giveup") for v in prefixes.values())
print("features", g.shape, "groups", counts)
print(f"branch points {len(prefixes)}, MIXED (both hack and giveup) {mixed}, matched pairs {pairs}")
norms = np.linalg.norm(g, axis=1)
print("norms: min %.4g median %.4g max %.4g" % (norms.min(), np.median(norms), norms.max()))
PY
