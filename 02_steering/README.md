# Qwen3-8B CoT steering

This experiment adds one normalized `diff` concept vector at one residual-stream
layer at a time. Positive alpha steers toward the named concept; negative alpha
steers toward its antagonist. Strength is measured as a fraction of the layer's
average residual-stream norm.

The default matrix is one shared baseline plus seven concepts, five layers, and
four nonzero strengths: 141 generations per question. Across all three
benchmarks that is 102,648 generations, so expect a months-long run even on four
V100s when outputs approach the context limit.

```bash
# Quick smoke test: one question, one concept, one layer, baseline + one strength
uv run python 02_steering/steer.py \
  --benchmark aime_2024 --limit 1 \
  --concept-pairs 367 --layers 22 --alphas 0,0.05

# Full four-GPU experiment
uv run python 02_steering/steer.py --benchmark all --num-workers 4

# Build aggregate tables plus reasoning-length and accuracy plots
uv run python 02_steering/summarize.py

# Lightweight implementation checks
uv run python 02_steering/test_steering.py
```

Results are appended under `02_steering/results/`. Re-running the same command
resumes completed condition/question pairs.
