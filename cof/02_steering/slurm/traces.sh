#!/bin/bash
# STEP 3. Replay of the saved traces: every concept is scored along every trace, which feeds
# the cardiogram and the ranking of concepts by correct-against-incorrect activation. Nothing
# is generated, so one card is enough and the pass takes well under an hour.
#
#   sbatch cof/02_steering/slurm/traces.sh
#   ALPHA=2 sbatch --export=ALL cof/02_steering/slurm/traces.sh
#   BENCHMARK=gpqa_diamond,math_500 TOKEN_ALPHAS=2,2.5 \
#     OUT=cof/02_steering/results/figures/combined \
#     sbatch --export=ALL cof/02_steering/slurm/traces.sh
#   INCLUDE_UNFINISHED=1 OUT=cof/02_steering/results/figures/include-unfinished \
#     sbatch --export=ALL cof/02_steering/slurm/traces.sh
#
# The CPU figures are built at the end of the same job, so the results are ready when it
# finishes. Rerunning is harmless: it overwrites its own outputs and touches nothing else.
#
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=traces_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Fail early with a clear Slurm error if the plotting environment is incomplete.
uv run python -c "import matplotlib, pandas, torch; print('plotting dependencies OK')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

VECTORS=~/korznikov_students/dm/exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic
RESULTS=${RESULTS:-cof/02_steering/results}
OUT=${OUT:-$RESULTS}
BENCHMARK=${BENCHMARK:-gpqa_diamond}
LAYER=${LAYER:-18}
# Which steering condition to replay; 0 is the unsteered baseline.
ALPHA=${ALPHA:-0}
# Lower it if the card runs out of memory on the longest traces.
CHUNK=${CHUNK:-1024}
# Optional subset for the token-saving chart. This is useful for a pooled plot, where only
# strengths present in every selected benchmark should be compared.
TOKEN_ALPHAS=${TOKEN_ALPHAS:-}
# Replay unclosed thinking spans too.  Their hidden states are not present in the old NPZ,
# so switching this on necessarily performs a fresh GPU replay.
INCLUDE_UNFINISHED=${INCLUDE_UNFINISHED:-0}

if [[ "$BENCHMARK" == *math_500* ]]; then
  # Older runs were written with correct=null when math-verify was absent.  The compact
  # override leaves the raw JSONL untouched and is read automatically by every plotter.
  uv run python cof/02_steering/rescore_math.py --results "$RESULTS"
fi

CARDIO_ARGS=()
TOKEN_ARGS=()
REASONING_SCOPE=closed
SCORE_SUFFIX=
if [[ "$INCLUDE_UNFINISHED" == 1 ]]; then
  CARDIO_ARGS+=(--include-unfinished)
  TOKEN_ARGS+=(--include-unfinished)
  REASONING_SCOPE=all
  SCORE_SUFFIX=-include-unfinished
fi

uv run python cof/02_steering/cardiogram.py \
  --results "$RESULTS" \
  --out "$OUT" \
  --vector-dir "$VECTORS" \
  --benchmark "$BENCHMARK" \
  --layer "$LAYER" \
  --alpha "$ALPHA" \
  --chunk-tokens "$CHUNK" \
  "${CARDIO_ARGS[@]}"

# The saved file is named after the strength the way Python prints it, so 2.0 becomes a2.
TAG=$(uv run python -c "import sys; print(format(float(sys.argv[1]), 'g'))" "$ALPHA")
SCORES=$OUT/concept-scores-L$LAYER-a$TAG$SCORE_SUFFIX.npz

# The ranking over every concept, which is what the figure in the report showed.
uv run python cof/02_steering/plot_concepts.py \
  --scores "$SCORES" \
  --vector-dir "$VECTORS" \
  --out "$OUT"

# The same ranking among the 16 CoT concepts alone. A smaller pool means a lower chance
# level, so a concept can clear it here that could not clear it against all 1036.
uv run python cof/02_steering/plot_concepts.py \
  --scores "$SCORES" \
  --vector-dir "$VECTORS" \
  --out "$OUT" \
  --pairs cot

if [[ -n "$TOKEN_ALPHAS" ]]; then
  TOKEN_ARGS+=(--alphas "$TOKEN_ALPHAS")
fi
uv run python cof/02_steering/plot_tokens.py \
  --results "$RESULTS" \
  --out "$OUT" \
  --benchmark "$BENCHMARK" \
  "${TOKEN_ARGS[@]}"

uv run python cof/02_steering/plot_outcomes.py \
  --results "$RESULTS" \
  --out "$OUT" \
  --benchmark "$BENCHMARK" \
  --vector-dir "$VECTORS" \
  --include-partial \
  --reasoning-scope "$REASONING_SCOPE"
