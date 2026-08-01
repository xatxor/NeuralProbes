#! /usr/bin/env bash
# One shard of multi-workload branch-point resampling.
#
# yds/fork.sh forks a single workload from a fixed directory. That is what produced a direction which
# reached 1.000 on the task it was fitted on and 0.154 on another -- with one workload in the corpus
# there is nothing in the objective that prefers the decision axis over the topic. This script forks
# SEVERAL workloads into one pool so the fit can be validated by holding an entire task out.
#
# Which workloads, and with what k, comes from forkplan.json rather than from here, because the
# choice depends on the measured unsteered hack rate: mixed branch points -- prefixes from which both
# a hack and a give-up were sampled -- are what the fit consumes, and their yield collapses as the
# base rate approaches zero. At p = 0.104 the first run got 40 mixed out of 275 branch points. Near
# p = 0.4 almost every branch point comes out mixed. So k is set per workload after the screen, not
# guessed before it.
set -u

SHARD=${1:?shard index}
SHARDS=${2:?shard count}
DTYPE=${DTYPE:-bfloat16}

ARCHIVE="forked.tar.gz"
LOG="shard.log"
: > "$LOG"
tar -czf "$ARCHIVE" --files-from /dev/null

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) mfork shard ${SHARD}/${SHARDS} dtype=${DTYPE} budget=${BUDGET:-9000}s"

export HF_HOME=/dev/shm/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
export AGENTIC_SANDBOX=subprocess
mkdir -p "$HF_HOME"

mkdir -p workloads
for module in common.py [0-9]*_*.py; do
  [ -f "$module" ] && mv "$module" workloads/
done

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

sleep $((SHARD * 90))
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

tar -xzf screened.tar.gz
# Split the flat screen output into one directory per variant: fork.py takes a single --dir and each
# entry in the plan names one workload/variant pair, so they must not be mixed.
python3 - <<'PY'
import json, pathlib, shutil
plan = json.load(open("forkplan.json"))
root = pathlib.Path("episodes")
for entry in plan["fork"]:
    variant = entry["variant"]
    target = pathlib.Path("sources") / variant
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in root.rglob("*.json"):
        if path.name.startswith("._"):                 # macOS AppleDouble sidecars match *.json
            continue
        if f"-{variant}-" in path.name:
            shutil.copy(path, target / path.name)
            moved += 1
    print(f"{variant:<22} {moved} source episodes")
PY

mkdir -p forked
started=$(date +%s)
BUDGET=${BUDGET:-9000}
python3 -c "
import json
for e in json.load(open('forkplan.json'))['fork']:
    print(e['workload'], e['variant'], e.get('k', 8), e.get('back', 3))
" > plan.txt

while read -r workload variant k back; do
  now=$(date +%s)
  if [ $((now - started)) -gt "$BUDGET" ]; then
    echo "$(date +%T) budget reached before ${variant}, stopping"
    break
  fi
  echo "$(date +%T) forking ${workload}/${variant} k=${k} back=${back}"
  python3 fork.py --dir "sources/${variant}" --out forked --workload "$workload" \
    --variant "$variant" --k "$k" --back "$back" \
    --shard "$SHARD" --shards "$SHARDS" --dtype "$DTYPE" --device cuda:0 \
    >> "fork-${variant}.log" 2>&1
  echo "$(date +%T) ${variant}: $(ls forked/*.json 2>/dev/null | wc -l) continuations so far"
done < plan.txt

# Stamp the workload onto every continuation. bipo.py's leave-one-workload-out CV groups on this
# field, and fork.py records the variant but not which module produced it.
python3 - <<'PY'
import glob, json
stamped = 0
for path in glob.glob("forked/*.json"):
    record = json.load(open(path))
    variant = record.get("variant") or ""
    if variant and not record.get("workload"):
        record["workload"] = variant
        json.dump(record, open(path, "w"))
        stamped += 1
print(f"stamped workload on {stamped} continuations")
PY

find forked -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null

# Gradients IN THE SAME CONTAINER. The model is already resident and the continuations are already
# on local disk, so running the feature stage here saves a second job's weight download and a full
# round trip through the Mac -- about twenty minutes per wave, which is the difference between
# finishing tonight and not. Each shard emits gradients for its own continuations; they are
# concatenated after the harvest.
echo "$(date +%T) gradients over $(ls forked/*.json 2>/dev/null | wc -l) continuations"
python3 bipo.py features --dir forked --out bipo --layer 18 --dtype "$DTYPE" \
  --window thinking --device cuda:0 >> "gradients.log" 2>&1
rc=$?
echo "$(date +%T) feature stage rc=${rc}; $(ls -la bipo/grads.npz 2>/dev/null | awk '{print $5}') bytes"
tail -5 gradients.log
# A missing declared output aborts the entire upload after the job has exited 0, so the file must
# exist even when the stage failed.
# Key names must match what the feature stage writes (g / meta / layer), or the merge silently sees
# an archive it cannot read and drops the shard.
mkdir -p bipo && [ -f bipo/grads.npz ] || python3 -c "
import numpy as np
np.savez_compressed('bipo/grads.npz', g=np.zeros((0, 4096), dtype=np.float32),
                    meta=np.array([], dtype=object), layer=np.array(18))"
cp bipo/grads.npz grads.npz

tar -czf "$ARCHIVE" forked
echo "$(date +%T) shard ${SHARD} complete: $(ls forked/*.json 2>/dev/null | wc -l) continuations"
echo "endings: $(grep -ho '"ending": "[a-z_]*"' forked/*.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
