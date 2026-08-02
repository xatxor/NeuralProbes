# OpenThoughts concept calibration

Estimate token-level cosine mean, variance, and standard deviation for all
1,036 `diff` concept vectors at Qwen3-8B layers 11, 14, 18, 22, and 25. The
script teacher-forces 100 ready reasoning traces from
`open-thoughts/OpenThoughts-114k`; it never generates answers.

```bash
uv run python 03_calibration/calibrate.py --num-workers 4
```

The seeded sample manifest and final `concept_stats.parquet`,
`normalization.npz`, and `run_metadata.json` are written under
`03_calibration/results/`. Statistics are pooled over every token inside the
`<think>...</think>` span, so longer traces contribute more observations.
