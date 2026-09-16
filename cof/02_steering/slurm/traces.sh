#!/bin/bash
# STEP 3. Replay of the saved traces: every concept is scored along every trace, which feeds
# the cardiogram and the ranking of concepts by correct-against-incorrect activation. Nothing
# is generated, so one card is enough and the pass takes well under an hour.
#
#   sbatch cof/02_steering/slurm/traces.sh
#   ALPHA=2 sbatch --export=ALL cof/02_steering/slurm/traces.sh
#
# The CPU figures are built at the end of the same job, so the results are ready when it
# finishes. Rerunning is harmless: it overwrites its own outputs and touches nothing else.
#
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=traces_%j.log

set -u
cd $SLURM_SUBMIT_DIR

export PYTORCH_ALLOC_CONF=expandable_segments:True

VECTORS=~/korznikov_students/dm/exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic
RESULTS=${RESULTS:-cof/02_steering/results}
BENCHMARK=${BENCHMARK:-gpqa_diamond}
LAYER=${LAYER:-18}
# Which steering condition to replay; 0 is the unsteered baseline.
ALPHA=${ALPHA:-0}
# Lower it if the card runs out of memory on the longest traces.
CHUNK=${CHUNK:-1024}

uv run python cof/02_steering/cardiogram.py \
  --results $RESULTS \
  --vector-dir $VECTORS \
  --benchmark $BENCHMARK \
  --layer $LAYER \
  --alpha $ALPHA \
  --chunk-tokens $CHUNK

# The saved file is named after the strength the way Python prints it, so 2.0 becomes a2.
TAG=$(uv run python -c "import sys; print(format(float(sys.argv[1]), 'g'))" $ALPHA)
SCORES=$RESULTS/concept-scores-L$LAYER-a$TAG.npz

# The ranking over every concept, which is what the figure in the report showed.
uv run python cof/02_steering/plot_concepts.py \
  --scores $SCORES \
  --vector-dir $VECTORS

# The same ranking among the 16 CoT concepts alone. A smaller pool means a lower chance
# level, so a concept can clear it here that could not clear it against all 1036.
uv run python cof/02_steering/plot_concepts.py \
  --scores $SCORES \
  --vector-dir $VECTORS \
  --pairs cot

uv run python cof/02_steering/plot_tokens.py \
  --results $RESULTS \
  --benchmark $BENCHMARK
