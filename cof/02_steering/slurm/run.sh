#!/bin/bash
# STEP 2. The real run: eight one-GPU jobs sharing the grid, each taking every eighth
# condition. Defaults already match the agreed design (layer 18, 16 CoT concepts plus
# joy, alpha +-1 to +-3, one baseline), so no grid flags are needed here.
#
#   sbatch cof/02_steering/slurm/run.sh
#
# Resubmit the same file after a time limit; it resumes from the shards already written.
# Set BATCH from what batchtest.sh measured.
#
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=steer_%A_%a.log

set -u
cd $SLURM_SUBMIT_DIR

# Recovers the memory PyTorch reserves but cannot reuse, which on a 32 GB card is several
# gigabytes and is the difference between a batch fitting and not.
export PYTORCH_ALLOC_CONF=expandable_segments:True

VECTORS=~/korznikov_students/dm/exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic
# On a 32 GB card the weights take ~16.4 GB and each sequence needs ~2.4 GB of KV cache at
# the 16k budget, so four fit and eight do not. A batch that still runs out is halved and
# retried automatically.
BATCH=${BATCH:-4}

uv run python cof/02_steering/steer.py \
  --benchmark gpqa_diamond \
  --num-workers 8 --worker-index $SLURM_ARRAY_TASK_ID \
  --batch-size $BATCH \
  --vector-dir $VECTORS
