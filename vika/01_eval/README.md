# Qwen3-8B thinking evaluation

Evaluates `Qwen/Qwen3-8B` with Qwen3 thinking explicitly enabled on:

- `HuggingFaceH4/aime_2024`
- `HuggingFaceH4/MATH-500`
- `Idavidrein/gpqa` (`gpqa_diamond`; gated—accept its Hugging Face conditions first)

## Setup

```bash
cd '/Users/olegchernikov/Mech Interp/AIRI_Summer_2026/Project'
uv add torch transformers accelerate math-verify pandas pyarrow safetensors huggingface-hub scipy umap-learn
```

For GPQA, authenticate first and accept the dataset conditions in the browser:

```bash
uv run hf auth login
```

The evaluator writes one JSONL record per example as it runs, so rerunning the same command resumes unfinished benchmarks. Results are saved under `results/`.

## Online loop detection

Generation now checks the unique 4-gram ratio in a sliding window of the latest 1,024 generated tokens. The check runs every 64 tokens, begins after at least 2,048 new tokens, and requires three consecutive windows below a unique-ratio threshold of `0.20`. This detects severe repetitive loops without changing logits, sampling, prompts, or concept activations.

When a loop is detected, generation deliberately continues for another 512 tokens before stopping. The retained tail preserves several repetitions for token-level concept analysis. The complete retained text and its FP16 traces are saved normally. Each JSONL record includes `loop_detected`, `loop_forced_stop`, and a `loop_detection` object with the trigger ratio and token offsets relative to the generated continuation.

Defaults can be adjusted without code changes:

```bash
uv run python 01_eval/evaluate.py --benchmark gpqa_diamond --concept-analysis \
  --loop-ngram-size 4 \
  --loop-window-tokens 1024 \
  --loop-unique-ratio-threshold 0.20 \
  --loop-check-every 64 \
  --loop-consecutive-windows 3 \
  --loop-min-new-tokens 2048 \
  --loop-extra-tokens 512
```

Use `--disable-loop-detection` to restore generation that continues until EOS or the model context limit.

## Useful options

```bash
# Smoke test
uv run python 01_eval/evaluate.py --benchmark aime_2024 --limit 3

# Smoke test with concept analysis
uv run python 01_eval/evaluate.py --benchmark gpqa_diamond --limit 1 --concept-analysis

# Run a single benchmark
uv run python 01_eval/evaluate.py --benchmark math_500

# Run one independent model replica on each of four visible GPUs
uv run python 01_eval/evaluate.py --benchmark math_500 --num-workers 4
```

## Concept-vector analysis

Add `--concept-analysis` to score the generated continuation against concept vectors from `josephofthebread/Qwen3-8B-concept-vectors`. By default the optimized evaluator records only the `diff` method on layers 18 and 22, for all 1,036 concept pairs. It stores one FP16 trace for the complete generated continuation; reasoning-only views are losslessly sliced from that same trace using `reasoning_start` and `reasoning_end`, so the reasoning trace is no longer duplicated on disk. Aggregate score tables retain the same reasoning-span `mean_cosine` definition. Token-highlight Parquet generation is disabled because the viewer reads complete per-token rankings directly from the dense trace.

The activation buffer defaults to 512 generated tokens. This changes only batching of the same per-token projection calculation. The complete projected method block is transferred from GPU to CPU once per selected layer and then split into method files on CPU. Model weights, concept vectors, projections, and traces remain FP16; only numerically sensitive reductions used to estimate means, variances, and geometry are accumulated in higher precision before being stored back in FP16.

Use these selectors when more methods or layers are required:

```bash
# Current optimized defaults: diff on layers 18 and 22
uv run python 01_eval/evaluate.py --benchmark all --num-workers 4 --concept-analysis

# Reproduce the former full analysis coverage
uv run python 01_eval/evaluate.py --benchmark all --num-workers 4 --concept-analysis \
  --concept-methods diff,concept_centered,antagonist_centered \
  --concept-layers 11,14,18,22,25
```

Use `--concept-pairs 12,74,985` to collect dense token-level cosine values only for selected pair IDs. Without this option, all 1,036 pairs are recorded. Use `--activation-chunk-size 128` to restore the old chunk size if needed; the default is now 512.

## Build and open the viewer

```bash
uv run python 01_eval/build_concept_report.py --results 01_eval/results
uv run python 01_eval/viewer_server.py --results 01_eval/results --port 8000
```

Open `http://127.0.0.1:8000/`.

Use `viewer_server.py`, not a plain `python -m http.server`: token traces, whole-response rankings, and per-token concept rankings are loaded through the viewer API.

The viewer contains four tabs:

1. **Response viewer**
   - a multi-select `class_name` filter with an **All classes** option; the concept selector is restricted to the selected feature classes;
   - a scope selector switches the token heatmap between the complete generated response and the reasoning span; new runs slice both views from one full-response trace, while older reasoning-only traces remain readable through fallback mode;
   - all concepts are ranked over the whole displayed response by mean token z-score or mean raw cosine, while a second list ranks all concepts at the selected token;
   - concepts from currently selected feature classes are shown first in both ranking lists, followed by concepts from other classes;
   - full-response token z-scores use full-response per-probe token baselines; reasoning traces retain their reasoning-token baselines;
   - concept choices are numbered in their current normalized or raw activation ranking;
   - pooled correlation with correctness.

2. **Probe analysis**
   - a Figure-5-style pairwise cosine-similarity heatmap for the `diff` probes from the selected fine-grained feature classes, ordered by their positions in the average-linkage hierarchy;
   - below it, a Figure-2-style heatmap of the selected probes × evaluation examples; correct examples are placed first, incorrect examples second, with a colored group header and separator;
   - per-probe example-level z-scores are shown by default, with raw mean cosine selectable;
   - dataset, layer, probe-method, value-scale, and multi-select feature-class controls, including **All classes**. Tick labels are shown only for a subset of rows and columns, while hover tooltips expose exact labels and values.

3. **Top concepts**
   - three horizontal ranked lists for all examples, correct examples, and incorrect examples of the selected dataset;
   - choose a single layer or average normalized values over whichever layers were recorded by the evaluator;
   - every scored concept from the selected feature classes is shown, not only the top 15;
   - each list scrolls independently while its shared horizontal axis remains visible below it;
   - four summaries can be selected: signed mean z-score `mean(z)`, positive mean z-score `mean(max(z, 0))`, negative mean z-score `mean(max(-z, 0))`, and raw mean cosine;
   - signed means measure directional group shifts and allow positive/negative deviations to cancel; positive means measure above-baseline movement toward the first concept-pair pole; negative means measure below-baseline movement toward the opposite pole as a positive magnitude;
   - all/correct/incorrect panels use one shared horizontal scale, and the multi-select feature-class filter includes **All classes**.

4. **Probe UMAP**
   - a full-page-width two-dimensional UMAP computed from normalized `diff` probe vectors for every recorded layer;
   - cosine distance with `n_neighbors=15`, `min_dist=0.1`, and a fixed random seed;
   - the original fine-grained `class_name` values are consolidated into ten manually defined semantic macro-clusters: reasoning, emotion/personality, relationships/communication, ethics/justice, politics/society/culture, AI identity/agency, alignment/safety/security, information/privacy/truth, work/planning/reliability, and creativity/values/worldviews;
   - translucent convex-hull fills mark the ten macro-cluster regions before points are drawn;
   - a conservative robust radial-plus-local-distance rule reassigns only strong spatial outliers to `Other` independently for each layer;
   - points are colored by their final UMAP cluster, the ten semantic cluster names are labeled on the plot, and hover shows the original fine-grained class and whether a point was reclassified as an outlier.

The Figure-2-style panel is an adaptation rather than a replication: benchmark examples are not controlled scenarios labeled to correspond to individual concepts, so a strong diagonal is not expected. The heatmap is intended for discovering which concepts activate together across the recorded examples.

The report builder stores heatmap matrices, the FP16 top-concept summary, FP16 UMAP coordinates, and `normalization-fp16.npz` under `results/concept_viewer/analysis/`. Per-probe baselines are pooled over all recorded datasets: reasoning-token and full-response-token statistics are stored separately, while example statistics come from each response’s reasoning-span mean cosine. The applied normalization is `z = (value - mean) / std`, separately for every concept, method, and layer. For the top-concept charts, each example-layer value is normalized first. The viewer can then aggregate `mean(z)`, `mean(max(z, 0))`, or `mean(max(-z, 0))` across the selected correctness group and all recorded layers; raw cosine remains available. The three panels share one horizontal scale. Heatmaps use compact quantized JSON; missing or zero-variance values use a dedicated sentinel and are shown in gray. Pairwise probe similarity and UMAP are not z-normalized, because they describe vector geometry rather than dataset activation magnitude.

`build_concept_report.py` treats the dense FP16 traces as the source of truth. If a resumed run left `concept_scores-*.parquet` incomplete, the builder reconstructs the missing reasoning-span `mean_cosine` rows before calculating mean/std. This prevents a one-example or partially overwritten Parquet file from producing zero-variance baselines and misleading zero z-scores. The evaluator also merges old and new score rows atomically when a run is resumed.

Raw `diff`-probe cosine similarities can naturally be below `0.1`; their absolute magnitude is not directly comparable across probes. Use standardized z-score for cross-concept ranking.

Thinking is enabled with `tokenizer.apply_chat_template(..., enable_thinking=True)` for every generation. Generation continues until the model emits EOS, subject to the model context window. The full output, including the `<think>...</think>` trace when emitted, is retained in JSONL together with its generation time. New records also store the raw benchmark prompt so analysis tooltips can identify examples; older compatible records remain readable and fall back to benchmark/example IDs.

AIME and GPQA use exact final-answer scoring. MATH-500 uses `math-verify` when installed; otherwise it records generations without claiming a score.

With `--num-workers N`, the parent process launches `N` subprocesses and assigns one visible GPU to each. Examples are distributed round-robin, each worker writes a separate resumable JSONL shard, and the parent atomically merges the shards into the usual benchmark JSONL and summary after every worker succeeds. Existing results are resumed only when their model, prompt fingerprint, concept-pair IDs, and trace files are compatible with the current run. A previously recorded superset is reusable: for example, old traces containing all three methods and all five layers satisfy a new `diff`/layers-18-and-22 run without recomputation. A run requesting methods or layers absent from an existing trace is recomputed.


The server refreshes the generated index.html from concept_viewer.html on startup.
