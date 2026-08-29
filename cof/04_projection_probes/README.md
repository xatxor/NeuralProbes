# Article-style projection probes

This pipeline replays saved AIME outputs through Qwen3-8B; it never generates
new answers. It removes OpenThoughts-math background PCs from the ready `diff`
concept vectors, then records raw residual-stream dot products.

```bash
python 04_projection_probes/pipeline.py fit-background \
  --sample-manifests \
    oleg/03_calibration/results/samples.jsonl \
    oleg/03_calibration/results_900/samples.jsonl \
  --output-dir 04_projection_probes/results/background \
  --num-workers 4

python 04_projection_probes/pipeline.py score-aime \
  --aime-results 01_eval_results/results_aime \
  --background-dir 04_projection_probes/results/background \
  --output-dir 01_eval_results/results_aime_projection \
  --num-workers 4

python 04_projection_probes/report.py \
  --results 01_eval_results/results_aime_projection

python 04_projection_probes/viewer_server.py \
  --results 01_eval_results/results_aime_projection \
  --port 18768
```

Open `http://127.0.0.1:18768/`. When the server is remote, forward the port
with `ssh -L 18768:127.0.0.1:18768 airi-summer`.
