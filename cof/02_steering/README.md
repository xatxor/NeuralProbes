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

The default matrix is one baseline plus the 16 CoT concepts and the `joy` control at
layer 18, across ten strengths (±1, ±1.5, ±2, ±2.5, ±3): 171 generations per question.
The 5 random sanity-check concepts (seed 2026, unrelated to reasoning) stay selectable
through `--concept-pairs` but are out of the default grid.

`--order` decides what an interrupted run leaves behind: `concept` (default) finishes
one concept's whole alpha curve before starting the next, `alpha` covers every concept
at one strength first. Both follow the order `--concept-pairs` and `--alphas` are given
in, so those lists set the priority.

```bash
# Quick smoke test: one question, one concept, baseline + one strength
uv run python cof/02_steering/steer.py \
  --benchmark gpqa_diamond --limit 1 \
  --concept-pairs 657 --alphas 1.5 --batch-size 1 \
  --vector-dir /path/to/Qwen_Qwen3-8B--assistant--semantic

# One concept's symmetric curve, coarse points first, on eight GPUs
uv run python cof/02_steering/steer.py \
  --benchmark gpqa_diamond --num-workers 8 --batch-size 8 \
  --concept-pairs 657 --alphas 1.0,-1.0,2.0,-2.0,3.0,-3.0,1.5,-1.5,2.5,-2.5 \
  --vector-dir /path/to/Qwen_Qwen3-8B--assistant--semantic

# Build aggregate tables plus reasoning-length and accuracy plots
uv run python 02_steering/summarize.py

# Lightweight implementation checks
uv run python 02_steering/test_steering.py
```

Questions inside one steering condition are generated together, `--batch-size` at a
time (default 8). Decoding reloads the weights every step regardless of how many
sequences are in flight, so a batch costs little more per step than a single sequence.
The KV cache is the limit: Qwen3-8B keeps about 144 KB per token, so a full 16k
generation needs roughly 2.5 GB per row on top of the 16 GB of weights.

Batching only groups rows that share a condition, since the injected vector belongs to
the condition. Padding goes on the left so the newest token of every row stays at the
position the steering hook writes to, and each row is cut at its own EOS afterwards.

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
