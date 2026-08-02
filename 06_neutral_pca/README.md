# Neutral-transcript PCA probes

Reproduce the neutral-PC removal used by Sofroniew et al. for Qwen3-8B
`diff` concept vectors, then score saved AdvBench baseline/GCG responses.

```bash
python 06_neutral_pca/pipeline.py neutral \
  --output 06_neutral_pca/results/alpaca-500

python 06_neutral_pca/pipeline.py score \
  --neutral 06_neutral_pca/results/alpaca-500 \
  --advbench 05_jailbreak_gcg/results/advbench-faster-gcg-all \
  --output 06_neutral_pca/results/advbench
```

Both commands use four visible GPUs. The neutral set is an unfiltered,
seeded sample of 500 `tatsu-lab/alpaca` training instructions. PCA is fitted
from a uniform reservoir of response-token residual activations independently
at each layer, retaining the minimum number of PCs that explain 50% variance.
