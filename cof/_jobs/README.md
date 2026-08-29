# DataSphere steering jobs

Set the DataSphere project ID and a Hugging Face token that can download
`Qwen/Qwen3-8B`, then submit a ten-way single-GPU sweep:

```bash
export DATASPHERE_PROJECT=<project-id>
export HF_TOKEN=<hugging-face-token>
./_jobs/deploy.sh
```

The launcher requires the `datasphere` CLI and `jq`; it reads the two variables
from either the environment or the repository's ignored `.env` file.

The launcher starts one `g2.1` job then forks it into ten jobs. Each receives
`--num-workers 10 --worker-index 0…9`, so every condition is generated exactly
once. Set `WORKERS=4` to use four GPUs instead.

With no arguments it runs the default Math-500 30-question sweep. Any
`steer.py` options override individual defaults, for example
`./_jobs/deploy.sh --limit 1` still runs Math-500.

Each completed job provides `steering-results.zip`. Download all ten with
`datasphere project job download-files --id <job-id>`, extract their contents
at the repository root, then run `uv run python 02_steering/summarize.py`.
The summarizer reads every `steering.worker-*.jsonl` shard automatically.
