# Qwen3-8B thinking evaluation

Evaluates `Qwen/Qwen3-8B` with Qwen3 thinking explicitly enabled on:

- `HuggingFaceH4/aime_2024`
- `HuggingFaceH4/MATH-500`
- `Idavidrein/gpqa` (`gpqa_diamond`; gated—accept its Hugging Face conditions first)

## Setup

```bash
cd '/Users/olegchernikov/Mech Interp/AIRI_Summer_2026/Project'
uv add torch transformers accelerate math-verify
uv run python 01_eval/evaluate.py --benchmark all
```

For GPQA, authenticate first (and accept the dataset's conditions in the browser):

```bash
uv run hf auth login
```

The script writes one JSONL record per example as it runs, so rerunning the same command resumes unfinished benchmarks. Results are saved under `results/`.

## Useful options

```bash
# Smoke test
uv run python 01_eval/evaluate.py --benchmark aime_2024 --limit 3

# Run a single benchmark
uv run python 01_eval/evaluate.py --benchmark math_500

# Run one independent model replica on each of four visible GPUs
uv run python 01_eval/evaluate.py --benchmark math_500 --num-workers 4
```

## Concept-vector analysis

Add `--concept-analysis` to score every reasoning token against all 1,036
concept pairs in `josephofthebread/Qwen3-8B-concept-vectors`. It uses only
layers 11, 14, 18, 22, and 25, and writes aggregate cosine scores plus the
three strongest positive and negative token highlights per method/layer.

Use `--concept-pairs 12,74,985` to score only selected pair IDs. Dense token
traces are stored as float16 NumPy matrices under
`results/traces/{benchmark}/{id}/`, one file per method/layer; aggregate scores
remain in Parquet.

```bash
uv run python 01_eval/evaluate.py --benchmark all --num-workers 4 --concept-analysis
uv run python 01_eval/build_concept_report.py
python 01_eval/viewer_server.py --results 01_eval/results
```

Open `http://localhost:8000/concept_viewer/`. The viewer compares `diff`,
`concept_centered`, and `antagonist_centered` side by side. Positive direction
is red and negative direction is blue; for `diff`, these are concept and
antagonist respectively.

Thinking is enabled with `tokenizer.apply_chat_template(..., enable_thinking=True)` for every generation. Generation continues until the model emits EOS, subject only to the model's context window. The full output, including the `<think>...</think>` trace when emitted by the model, is retained in the JSONL file together with its generation time. AIME and GPQA use exact final-answer scoring. MATH-500 uses `math-verify` when installed; otherwise it records generations without claiming a score.

With `--num-workers N`, the parent process launches `N` subprocesses and
assigns one visible GPU to each. Examples are distributed round-robin, each
worker writes a separate resumable JSONL shard, and the parent atomically
merges the shards into the usual benchmark JSONL and summary after every
worker succeeds. Existing results are resumed only when their model and prompt
fingerprint match the current run.
