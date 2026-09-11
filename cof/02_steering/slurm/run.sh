#!/bin/bash
# STEP 2. The real run: a job array of one-GPU tasks sharing the grid, each taking every Nth
# condition. Layer 18, all ten alphas (+-1 to +-3) and the baseline are steer.py defaults;
# the concepts are chosen here.
#
#   sbatch cof/02_steering/slurm/run.sh
#
# Resubmitting the same file is harmless: it resumes from every shard already written,
# including ones left by an earlier array of a different size.
#
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14-00:00:00
#SBATCH --array=0-6
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
# With --order concept each concept's whole alpha curve finishes before the next starts:
# planning first, the joy control early, then honest admission (the specificity check), then
# proof-style.
CONCEPTS=${CONCEPTS:-657,532,459,703}
# The worker count has to match the array size; SLURM exports it for array jobs.
WORKERS=${SLURM_ARRAY_TASK_COUNT:-7}

uv run python cof/02_steering/steer.py \
  --benchmark gpqa_diamond \
  --concept-pairs $CONCEPTS \
  --num-workers $WORKERS --worker-index $SLURM_ARRAY_TASK_ID \
  --batch-size $BATCH \
  --vector-dir $VECTORS
