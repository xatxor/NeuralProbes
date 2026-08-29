# Qwen3 GCG jailbreak concepts

This safety-research pipeline trains a universal Faster-GCG suffix on 20 AdvBench prompt/target pairs, then compares baseline and attacked responses on 100 held-out AdvBench prompts. Targets are AdvBench's instruction-specific affirmative prefixes. Thinking is disabled and all generated response tokens are scored against `diff` concept vectors.

```bash
python 05_jailbreak_gcg/pipeline.py prepare
python 05_jailbreak_gcg/pipeline.py attack
python 05_jailbreak_gcg/pipeline.py generate --num-workers 1 --worker-index 0
pip install git+https://github.com/dsbowen/strong_reject.git@main
HF_TOKEN=... python 05_jailbreak_gcg/pipeline.py judge
python 05_jailbreak_gcg/pipeline.py report
python 05_jailbreak_gcg/viewer_server.py --results 05_jailbreak_gcg/results/beavertails-gcg --port 8000
```

For four generation workers, launch the `generate` command once per worker with its own `CUDA_VISIBLE_DEVICES` and `--worker-index 0..3 --num-workers 4`. Do not run attack workers concurrently: universal GCG is one optimization trajectory.

The complete detached run can be launched with `STEPS=200 nohup bash 05_jailbreak_gcg/run_full.sh > logs/gcg-full.log 2>&1 &`.

The default GCG configuration is 500 steps, a 20-token suffix, 512 candidates, top-256 gradient candidates, and 32-candidate GPU chunks. A smoke run is:

```bash
python 05_jailbreak_gcg/pipeline.py all --train-samples 1 --test-samples 1 --steps 1 --batch-size 4 --topk 4 --candidate-chunk-size 2
```

Results include raw harmful model outputs. They are intentionally kept under the gitignored `results/` tree.
