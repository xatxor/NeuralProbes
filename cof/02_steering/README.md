# Qwen3-8B CoT steering

This experiment adds one raw `diff` concept vector at one residual-stream
layer at a time, unscaled. Positive alpha steers toward the named concept;
negative alpha steers toward its antagonist. Strength is measured in multiples
of the concept's own difference-vector norm at that layer (`alpha=1` adds a
vector exactly as large as the pair's natural `diff` vector) — not a fraction
of the residual-stream norm.

Vectors are read from a local directory (`--vector-dir`), not Hugging Face
Hub: `manifest.json`, `diff.safetensors`, `pairs.parquet` produced by the
probes pipeline. Pass the directory that matches the presentation you want
(e.g. `Qwen_Qwen3-8B--assistant--semantic`).

The default matrix is three baseline repeats plus 16 CoT concepts, the `joy`
control, and 5 random sanity-check concepts (seed 2026, unrelated to
reasoning), layers 11/14/18/22/25, and sixteen nonzero strengths (±1.5, ±2,
±2.5, ±3, ±3.5, ±4, ±4.5, ±5): 1,763 generations per question. This is a lot —
run a small pilot before committing to the full grid on a full dataset.

```bash
# Quick smoke test: one question, one concept, one layer, baseline + one strength
uv run python 02_steering/steer.py \
  --benchmark aime_2024 --limit 1 --baseline-repeats 1 \
  --concept-pairs 367 --layers 18 --alphas 1.5 \
  --vector-dir /path/to/Qwen_Qwen3-8B--assistant--semantic

# Full four-GPU experiment
uv run python 02_steering/steer.py --benchmark all --num-workers 4 \
  --vector-dir /path/to/Qwen_Qwen3-8B--assistant--semantic

# Build aggregate tables plus reasoning-length and accuracy plots
uv run python 02_steering/summarize.py

# Lightweight implementation checks
uv run python 02_steering/test_steering.py
```

Generation is bounded by `--max-new-tokens` (default 16,384). Strong steering
often never terminates: at alpha 5 a single unbounded generation can fill the
whole 41k context and cost roughly two hours, against about three minutes for
an unsteered baseline. Records carry `hit_token_budget` so a truncated run is
distinguishable from a wrong answer.

An online loop detector (`UniqueNGramLoopDetector`, shared with the eval
harness) stops sustained low-diversity repetition and records `loop_detected`.
Disable it with `--disable-loop-detection`.

Results are appended under `02_steering/results/`. Re-running the same command
resumes completed condition/question pairs; a change of `--vector-dir` (or any
vector recapture, tracked via the manifest's `capture_merge_key`) invalidates
old records automatically rather than silently reusing them. Lowering
`--max-new-tokens` only invalidates records that would have been truncated by
the new budget; generations that stopped on their own inside it are reused.

Summaries use the three alpha-zero generations per question as the baseline;
override them with `--baseline-repeats`. Pair `532` (`joy`, versus `sadness`)
is the default non-CoT control vector.
