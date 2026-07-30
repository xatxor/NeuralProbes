#! /usr/bin/env bash
# One shard of the steering dose-response sweep for an extracted direction.
#
# The question is causal, so the arms are built to fail rather than to pass:
#
#   add +/-alpha    does the direction move the outcome, and monotonically in alpha?
#   project         does REMOVING it change anything? A direction that steers but whose removal does
#                   nothing is a lever the experimenter found, not a mechanism the model uses.
#   randomN         seeded random directions at the SAME norm. This project has already measured that
#                   steering any unit direction at these layers moves generation length, so a random
#                   arm is not decoration -- it is the thing the real arm has to beat.
#   shared          the normalised mean of all 1036 concepts, which catches an effect belonging to
#                   the set's shared component rather than to any direction in particular.
#
# alpha = 0 is deliberately absent: 288 unsteered gate episodes already exist and are the control
# arm. Re-running them would spend a GPU-hour reproducing a number we have.
set -u

SHARD=${1:?shard index}
SHARDS=${2:?shard count}
SEEDS=${3:-48}          # seeds per arm
SEED_FROM=${4:-1000}    # far from the gate run's 0..287, so no episode is a repeat
DTYPE=${DTYPE:-bfloat16}
LAYER=${LAYER:-18}

ARCHIVE="episodes.tar.gz"
LOG="shard.log"
: > "$LOG"
tar -czf "$ARCHIVE" --files-from /dev/null

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) steer shard ${SHARD}/${SHARDS} seeds=${SEEDS} from=${SEED_FROM} layer=${LAYER} dtype=${DTYPE}"

export HF_HOME=/dev/shm/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
export AGENTIC_SANDBOX=subprocess
mkdir -p "$HF_HOME"

mkdir -p workloads
for module in common.py 0*_*.py; do
  [ -f "$module" ] && mv "$module" workloads/
done

tar -xzf vectors.tar.gz
tar -xzf episodes-gate.tar.gz
# alpha is a fraction of the residual norm, which agent.py measures on a real trajectory rather than
# assuming. Any gate episode serves; fixing it by name keeps every arm on the same scale.
NORMFROM="episodes/gate/impossible_tests-gate-seed0.json"
echo "$(date +%T) vectors: $(ls vectors/ | tr '\n' ' ')"

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 - <<'PY'
import timeit
data = list(range(100000))
best = min(timeit.repeat(lambda: sum(data), number=10, repeat=20)) / 10
print("sum(100k) %.6fs vs 0.000100s target: %.1fx too slow%s"
      % (best, best / 0.0001, "" if best > 0.0001 else "  *** TARGET IS REACHABLE, STOP ***"))
PY

sleep $((SHARD * 45))
for attempt in 1 2 3 4 5; do
  python3 - <<'PY' && break
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("weights present")
PY
  echo "$(date +%T) weight fetch attempt ${attempt} failed, backing off"
  sleep $((attempt * 60))
done

# Arms as "direction mode alpha". Every shard builds the identical list and takes its stride, so no
# arm/seed combination is skipped or run twice.
python3 - "$SHARD" "$SHARDS" "$SEEDS" "$SEED_FROM" > jobs.txt <<'PY'
import sys
shard, shards, count, start = (int(a) for a in sys.argv[1:5])
extracted = "file:vectors/vector-L18-mean.npy"
arms = [(extracted, "add", a) for a in (-0.10, -0.05, 0.05, 0.10)]
arms += [(extracted, "project", 0.0)]
arms += [("random0", "add", -0.10), ("random0", "add", 0.10)]
arms += [("shared", "add", 0.10)]
jobs = [(d, m, a, s) for (d, m, a) in arms for s in range(start, start + count)]
for index, job in enumerate(jobs):
    if index % shards == shard:
        print(*job)
PY
echo "$(date +%T) this shard runs $(wc -l < jobs.txt) episodes across $(cut -d' ' -f1-3 jobs.txt | sort -u | wc -l) arms"

mkdir -p episodes
while read -r direction mode alpha seed; do
  echo "$(date +%T) start ${direction} ${mode} a=${alpha} seed${seed}"
  python3 agent.py --workload 01_impossible_tests --variant gate --seed "$seed" \
    --dtype "$DTYPE" --direction "$direction" --mode "$mode" --alpha "$alpha" \
    --steer-layer "$LAYER" --norm-from "$NORMFROM" --out episodes \
    > "episodes/arm-$(echo "$direction" | tr '/:' '__')-${mode}-a${alpha}-seed${seed}.log" 2>&1
  echo "$(date +%T) done  rc=$?"
done < jobs.txt

tar -czf "$ARCHIVE" episodes
echo "$(date +%T) shard ${SHARD} complete: $(ls episodes/*.json 2>/dev/null | wc -l) episodes"
echo "endings: $(grep -ho '"ending": "[a-z_]*"' episodes/*.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
