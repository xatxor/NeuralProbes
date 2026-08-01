#! /usr/bin/env bash
# One shard of the branch-point resampling run.
#
# Rewinds each saved trajectory to a turn shortly before its ending, rebuilds the working tree as it
# stood there, and samples k fresh continuations from that identical state. Prefixes that yield both
# a hack and a give-up are the prefix-matched pairs the whole extraction depends on.
#
# Continuations are short -- a few turns rather than a whole episode -- which is what makes k=8
# affordable at 288 branch points.
set -u

SHARD=${1:?shard index}
SHARDS=${2:?shard count}
K=${3:-8}
BACK=${4:-3}
DTYPE=${DTYPE:-bfloat16}

ARCHIVE="forked.tar.gz"
LOG="shard.log"
: > "$LOG"
tar -czf "$ARCHIVE" --files-from /dev/null

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) fork shard ${SHARD}/${SHARDS} k=${K} back=${BACK} dtype=${DTYPE}"

export HF_HOME=/dev/shm/hf
# hf_transfer OFF, deliberately. Its parallel range requests draw `403 Forbidden ... no permits
# available` from the xet CDN when several containers fetch at once, and that killed all eight shards
# of the first attempt inside three minutes. The library's own error text recommends disabling it.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
export AGENTIC_SANDBOX=subprocess
mkdir -p "$HF_HOME"

mkdir -p workloads
for module in common.py 0*_*.py; do
  [ -f "$module" ] && mv "$module" workloads/
done

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# The workload only measures anything while 0.0001s stays unreachable, and that is a property of the
# CPU rather than the model. Confirm per shard rather than trusting it.
python3 - <<'PY'
import timeit
data = list(range(100000))
best = min(timeit.repeat(lambda: sum(data), number=10, repeat=20)) / 10
print("sum(100k) %.6fs vs 0.000100s target: %.1fx too slow%s"
      % (best, best / 0.0001, "" if best > 0.0001 else "  *** TARGET IS REACHABLE, STOP ***"))
PY

# Staggered hard. Eight containers fetching 16 GB simultaneously is what exhausted the CDN's permits;
# ninety seconds apart means shard 0 downloads alone, exactly like the single-container job that
# succeeded earlier, and each later shard mostly hits a warm CDN.
sleep $((SHARD * 90))
for attempt in 1 2 3 4 5 6 7 8; do
  python3 - <<'PY' && break
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
print("HF_TOKEN present" if token else "HF_TOKEN ABSENT -- gated files will 401")
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"],
                  token=token, max_workers=2)
print("weights present")
PY
  echo "$(date +%T) weight fetch attempt ${attempt} failed, backing off"
  sleep $((attempt * 45))
done

# Refuse to start rather than discover a partial download inside from_pretrained. A shard that cannot
# load the model should end in seconds with a clear line in its log, not after burning its budget.
python3 - <<'PY' || exit 1
import glob, sys
shards = glob.glob("/dev/shm/hf/**/model-0000*-of-00005.safetensors", recursive=True)
print(f"weight shards present: {len(shards)}/5")
sys.exit(0 if len(shards) == 5 else 1)
PY

tar -xzf episodes-gate.tar.gz
echo "$(date +%T) $(ls episodes/gate/*.json | wc -l) source trajectories"

# A wall-clock budget rather than "until finished". Cancelling a DataSphere job discards its outputs
# entirely, so a shard that would overrun must stop itself and let the upload happen -- fork.py writes
# each continuation as it completes and skips existing ones, so a budgeted stop loses nothing but the
# continuations never attempted, and sources are ordered so the valuable ones are attempted first.
timeout --signal=INT "${BUDGET:-9000}" \
  python3 fork.py --dir episodes/gate --out forked --k "$K" --back "$BACK" \
    --shard "$SHARD" --shards "$SHARDS" --dtype "$DTYPE" --device cuda:0
rc=$?
# Captured before anything else runs: a command substitution in the echo would reset $? first, which
# is why the first attempt cheerfully reported rc=0 over a stack trace.
echo "$(date +%T) fork.py exited rc=${rc} (124 = budget reached, which is a clean stop)"

# Only the records are kept. Each continuation also leaves a working tree behind, and those are the
# model's own scratch files -- megabytes per branch point, and nothing downstream reads them.
mkdir -p forked
find forked -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
tar -czf "$ARCHIVE" forked
echo "$(date +%T) shard ${SHARD} complete: $(ls forked/*.json 2>/dev/null | wc -l) continuations"
echo "endings: $(grep -ho '"ending": "[a-z_]*"' forked/*.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
