#! /usr/bin/env bash
# One shard of the steering evaluation, driven by arms.json.
#
# The arms and their justifications live in agentic/arms.py, not here, so that what got a GPU-hour is
# auditable in one place. This script only expands (arm x seed) across shards and runs them in the
# order arms.py emitted -- which is priority order, so a budget cut costs the tail rather than the
# head-to-head.
#
# Every arm is measured against the same seeds. Steering comparisons across different seed sets would
# confound the direction with the sample, and at a ~10% base rate that confound is larger than any
# effect worth reporting.
set -u

SHARD=${1:?shard index}
SHARDS=${2:?shard count}
LAYER=${3:-18}
DTYPE=${DTYPE:-bfloat16}

ARCHIVE="episodes.tar.gz"
LOG="shard.log"
: > "$LOG"
tar -czf "$ARCHIVE" --files-from /dev/null

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) sweep shard ${SHARD}/${SHARDS} layer=${LAYER} dtype=${DTYPE} budget=${BUDGET:-10800}s"

export HF_HOME=/dev/shm/hf
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
python3 - <<'PY'
import timeit
data = list(range(100000))
best = min(timeit.repeat(lambda: sum(data), number=10, repeat=20)) / 10
print("sum(100k) %.6fs vs 0.000100s target: %.1fx too slow%s"
      % (best, best / 0.0001, "" if best > 0.0001 else "  *** TARGET IS REACHABLE, STOP ***"))
PY

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

tar -xzf vectors.tar.gz
tar -xzf episodes-gate.tar.gz
echo "$(date +%T) vectors: $(ls vectors/ | tr '\n' ' ')"
# alpha is a fraction of the residual norm, measured on real agentic text at the steering layer
# rather than assumed. Fixing the source episode keeps every arm on one scale.
NORMFROM="episodes/gate/impossible_tests-gate-seed0.json"

python3 - "$SHARD" "$SHARDS" > jobs.txt <<'PY'
import json, sys
shard, shards = int(sys.argv[1]), int(sys.argv[2])
plan = json.load(open("arms.json"))
start, count = plan["seed_from"], plan["seeds"]
# Per-cell seed counts when the plan supplies them: cheap arms get fewer episodes, which matters
# when every episode is a transcript someone has to read.
work = [(j["name"], j["direction"], j["mode"], j["alpha"], s)
        for j in plan["jobs"]
        for s in range(start, start + int(j.get("seeds", count)))]
for index, item in enumerate(work):
    if index % shards == shard:
        print(*item)
PY
echo "$(date +%T) this shard has $(wc -l < jobs.txt) episodes queued"

mkdir -p episodes
started=$(date +%s)
BUDGET=${BUDGET:-10800}
while read -r name direction mode alpha seed; do
  now=$(date +%s)
  if [ $((now - started)) -gt "$BUDGET" ]; then
    echo "$(date +%T) budget reached, stopping with $(ls episodes/*.json 2>/dev/null | wc -l) done"
    break
  fi
  echo "$(date +%T) ${name} ${mode} a=${alpha} seed${seed}"
  python3 agent.py --workload 01_impossible_tests --variant gate --seed "$seed" \
    --dtype "$DTYPE" --direction "$direction" --mode "$mode" --alpha "$alpha" \
    --steer-layer "$LAYER" --norm-from "$NORMFROM" --out episodes \
    > "episodes/${name}-${mode}-a${alpha}-seed${seed}.log" 2>&1
  # The arm name is not recoverable from agent.py's own record when two arms share a direction, so
  # it is stamped onto the episode here.
  python3 - "$name" "$alpha" "$mode" "$seed" <<'PY'
import glob, json, sys
name, alpha, mode, seed = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
for path in glob.glob(f"episodes/*seed{seed}.json"):
    record = json.load(open(path))
    if record.get("arm"):
        continue
    record["arm"] = {"name": name, "alpha": alpha, "mode": mode}
    json.dump(record, open(path, "w"))
    break
PY
done < jobs.txt

find episodes -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
tar -czf "$ARCHIVE" episodes
echo "$(date +%T) shard ${SHARD} complete: $(ls episodes/*.json 2>/dev/null | wc -l) episodes"
echo "endings: $(grep -ho '"ending": "[a-z_]*"' episodes/*.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
