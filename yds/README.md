# Yandex DataSphere jobs
Everything needed to run the GPU stages on DataSphere:
a `*.yaml` spec per stage,
the `*.sh` container entrypoint beside it,
and the tooling that submits and retrieves them.

## General information
- Submit a wave and collect it:
    ```bash
    CONFIG=yds/gen.yaml OUT=.bak/gen yds/deploy.sh
    LOG=.bak/gen-deploy.log yds/watch.sh
    ```
- Script reference:
    | Name | Description |
    | - | - |
    | [deploy.sh](./deploy.sh) | Submit shard 0, fork the rest, poll to completion, download. Driven by `CONFIG` and `OUT` |
    | [chain.sh](./chain.sh) | Run `SMOKE` end to end first, launch `CONFIG` only if it succeeded |
    | [watch.sh](./watch.sh) | Print a status table for a running deployment; safe alongside `deploy.sh` |
    | [requirements.txt](./requirements.txt) | Installed inside the container; separate from `pyproject.toml` groups |
- Job reference:
    | Name | Entrypoint | Description |
    | - | - | - |
    | [stats.yaml](./stats.yaml) | — | Sweep `feature_stories` and accumulate sums, counts and second moments. |
    | [gen.yaml](./gen.yaml) | [gen.sh](./gen.sh) | Phase 0 wave A: generate ±α pairs for 1101 directions. Qwen3-8B only |
    | [judge.yaml](./judge.yaml) | [judgerun.sh](./judgerun.sh) | Phase 0 wave B: blinded pairwise judging. Gemma 3 12B, on disk |
    | [translate.yaml](./translate.yaml) | [translate.sh](./translate.sh) | Russian names for concepts and ontology classes, for the viewer |
