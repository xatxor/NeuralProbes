# Individual GCG + assistant-token steering

This creates 100 fixed AdvBench rows (the same rows optimize and evaluate their
own suffixes) and 100 deterministic Alpaca rows.  It has five unsteered
generation conditions: AdvBench baseline/GCG/random, Alpaca baseline/random.
It then steers each condition at layers 18 and 25 with pair 272, `detecting
steganographic intent`, at `-0.25` and `0.25`.

The steering modes are:

- `assistant`: only Qwen's literal `assistant` boundary token during prefill.
- `assistant_and_generated`: that boundary token plus each subsequent cached
  generated-token forward pass.

Nothing has been run by this implementation.  A four-GPU run later would be:

```bash
cd 07_individual_gcg
python -u experiment.py all --num-workers 4
```

Each stage is resumable.  Outputs go to `results/advbench-100-individual-gcg/`:
per-prompt attacks, per-worker JSONL responses, StrongREJECT judgments for
AdvBench only, and `summary.csv`.
