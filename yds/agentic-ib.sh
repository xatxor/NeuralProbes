#! /usr/bin/env bash
# One shard of the ImpossibleBench validation: does the reward-hacking direction fitted on workload 01
# fire on tasks it has never seen?
#
# Two splits of the same 20 problems. `conflicting` carries assertions that contradict the docstring,
# so passing means violating the spec. `original` is the identical problem, solvable, and is the
# negative control -- without it, "the direction fires on ImpossibleBench" cannot be told apart from
# "the direction fires on coding tasks". The pairing is per problem, not merely per split.
#
# The window vectors are computed HERE rather than back on a V100, because these episodes are
# generated in bfloat16 and the V100s have none. Replaying elsewhere in fp16 would read a residual
# stream the generation never had.
#
# No Docker: not installed, docker-in-docker impossible, unshare blocked by seccomp -- all measured.
# The job container is itself ephemeral and single-tenant, so it is the isolation boundary.
set -u

SHARD=${1:?shard index}
SHARDS=${2:?shard count}
SPLITS=${3:-conflicting,original}
TASKS=${4:-20}          # first N task ids, the same ids in both splits
SEEDS=${5:-4}
DTYPE=${DTYPE:-bfloat16}

# Declared outputs must exist before anything can go wrong: a missing one aborts the whole upload and
# takes the files that do exist with it. Static names -- `${SHARD}` is not substituted in an
# `outputs:` block, which once cost a complete run.
ARCHIVE="episodes.tar.gz"
WINDOWS="windows.npz"
LOG="shard.log"
: > "$LOG"
tar -czf "$ARCHIVE" --files-from /dev/null
python3 -c "import numpy; numpy.savez('$WINDOWS', empty=numpy.zeros(1))"

exec > >(tee -a "$LOG") 2>&1
echo "$(date +%T) shard ${SHARD}/${SHARDS} splits=${SPLITS} tasks=${TASKS} seeds=${SEEDS} dtype=${DTYPE}"

export HF_HOME=/dev/shm/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
export AGENTIC_SANDBOX=subprocess
mkdir -p "$HF_HOME"

# local-paths flattens everything into the job directory; agent.py imports workloads as a package.
mkdir -p workloads
for module in common.py [0-9]*_*.py; do
  [ -f "$module" ] && mv "$module" workloads/
done
echo "workloads present: $(ls workloads/ 2>/dev/null | tr '\n' ' ')"

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, 'bf16', torch.cuda.is_bf16_supported())"

# Ten containers pulling 16 GB at once once killed seven of ten shards. Stagger, then retry.
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

# The task list is taken from the dataset itself and intersected across splits, so a task that exists
# in one split but not the other cannot produce an unpaired episode. Every shard computes the same
# full list and takes its stride, so no shard can skip or duplicate another's work.
python3 - "$SHARD" "$SHARDS" "$SPLITS" "$TASKS" "$SEEDS" > jobs.txt <<'PY'
import sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

shard, shards = int(sys.argv[1]), int(sys.argv[2])
splits = sys.argv[3].split(",")
want, seeds = int(sys.argv[4]), int(sys.argv[5])

ids = {}
for split in splits:
    path = hf_hub_download("fjzzq2002/impossible_livecodebench",
                           f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    ids[split] = [row["task_id"] for row in pq.read_table(path).to_pylist()]

shared = [t for t in ids[splits[0]] if all(t in ids[s] for s in splits)][:want]
print(f"# {len(shared)} paired tasks across {splits}", file=sys.stderr)
if len(shared) < want:
    print(f"# WARNING: only {len(shared)} of {want} tasks exist in every split", file=sys.stderr)

jobs = [(s, t, d) for s in splits for t in shared for d in range(seeds)]
for index, (split, task, seed) in enumerate(jobs):
    if index % shards == shard:
        print(split, task, seed)
PY
echo "$(date +%T) this shard runs $(wc -l < jobs.txt) episodes"

mkdir -p episodes
while read -r split task seed; do
  echo "$(date +%T) start ${split}/${task} seed${seed}"
  IMPOSSIBLE_SPLIT="$split" IMPOSSIBLE_TASK="$task" \
    python3 agent.py --workload 10_impossiblebench --variant gate --seed "$seed" \
      --dtype "$DTYPE" --out episodes > "episodes/${split}-${task}-seed${seed}.log" 2>&1
  echo "$(date +%T) done  ${split}/${task} seed${seed} rc=$?"
done < jobs.txt

echo "$(date +%T) endings: $(grep -ho '"ending": "[a-z_]*"' episodes/*.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"

# Read the residual stream in the SAME dtype the episodes were generated in. This is the whole reason
# the window stage runs on the A100 instead of back on a V100.
python3 extract.py --stage window --dir episodes --out . --dtype "$DTYPE" --shard 0 --shards 1
[ -f windows-0.npz ] && mv -f windows-0.npz "$WINDOWS"

tar -czf "$ARCHIVE" episodes
echo "$(date +%T) shard ${SHARD} complete: $(ls episodes/*.json 2>/dev/null | wc -l) episodes, $(du -sh "$ARCHIVE" | cut -f1)"
