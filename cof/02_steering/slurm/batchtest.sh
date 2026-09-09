#!/bin/bash
# STEP 1. Throughput check. Same eight questions and the same condition at three batch
# sizes, each into its own results directory so none of them resumes from another.
# Compare the tok/s it prints against the 17 tok/s of the unbatched pilot, then use the
# largest size that did not run out of memory for the real run.
#
#   sbatch cof/02_steering/slurm/batchtest.sh
#
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=batchtest_%j.log

set -u
cd $SLURM_SUBMIT_DIR

VECTORS=~/korznikov_students/dm/exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic
OUT=cof/02_steering/results-batchtest

nvidia-smi --query-gpu=name,memory.total --format=csv

for SIZE in 1 8 16; do
  echo "=== batch-size $SIZE ==="
  uv run python cof/02_steering/steer.py \
    --benchmark gpqa_diamond --limit 8 \
    --concept-pairs 657 --alphas 1.5 \
    --batch-size $SIZE \
    --results-dir $OUT/size-$SIZE \
    --vector-dir $VECTORS || echo "batch-size $SIZE failed (out of memory?)"
done

echo
echo "=== throughput ==="
uv run python - "$OUT" <<'PY'
import json, sys
from pathlib import Path

for directory in sorted(Path(sys.argv[1]).glob("size-*"), key=lambda p: int(p.name.split("-")[1])):
    records = {}
    for shard in directory.glob("steering*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["key"]] = record
    if not records:
        print(f"{directory.name}: no records (failed?)")
        continue
    tokens = sum(r["generated_token_count"] for r in records.values())
    # generation_seconds is each row's share of its batch, so the sum is the wall time.
    seconds = sum(r["generation_seconds"] for r in records.values())
    print(f"{directory.name}: {len(records)} generations, {tokens:,} tokens, {seconds:,.0f} s -> {tokens / seconds:.1f} tok/s")
PY
