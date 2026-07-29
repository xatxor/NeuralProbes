#! /usr/bin/env bash
# Prove the judging box works and measure what the real run will cost.
#
# This runs judge.py itself, unmodified, on one thin slice: --shards is set high so shard 0 is a
# small fraction of the comparisons. Nothing here duplicates the judge's prompt building or schema,
# so what is measured is exactly what will run.
#
# It answers three things before both cards are committed to ~168,000 verdicts: that the model loads
# on the attached hardware, that verdicts parse into the eight-field schema, and the real throughput
# including CUDA graph capture.
#
# gpt-oss answers in harmony format: a long "analysis" channel, then "final" with the JSON. A tight
# token cap truncates mid-analysis and the JSON is never written, which looks like a parse failure
# rather than a budget one -- 245 of 248 verdicts were lost that way at 512 tokens.
#
# Usage:  ./judgebox-smoke.sh [SLICE] [EFFORT] [REPLY_TOKENS]
#         COMPARISONS=plus_baseline,minus_baseline ./judgebox-smoke.sh
#         SLICE 500 judges roughly 1/500th of the comparisons.
#
# COMPARISONS must match what the real run will use, or the throughput measured here is for a
# different set of comparisons than the one being committed to.
set -uo pipefail

slice=${1:-500}
effort=${2:-medium}
tokens=${3:-2048}
comparisons=${COMPARISONS:-plus_baseline,minus_baseline}
# 0.92 leaves ~11 GiB free on an H200, which is roughly what a layer-truncated Qwen3-8B needs.
# Above ~0.95 nothing else fits on the card, and graph capture and peak activations come out of
# the same headroom, so too high aborts partway rather than running slower.
util=${GPU_MEM_UTIL:-0.97}
root=${ROOT:-$HOME/judge}
venv="$root/venv"
model=${MODEL:-openai/gpt-oss-120b}

cd "$root" || exit 1

echo "=== cards ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4
nvidia-smi >/dev/null 2>&1 || { echo "no working driver; nvidia-smi cannot talk to the GPUs" >&2; exit 1; }

rows=$(cat runs/shard-*/runs.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "generations available: ${rows:-0}"
echo "comparisons: $comparisons, gpu_memory_utilization: $util"
[ "${rows:-0}" -gt 0 ] || { echo "no generations under runs/; upload them first" >&2; exit 1; }

mkdir -p smoke
started=$(date +%s)

CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 \
    "$venv/bin/python" judge.py \
    --mode rescreen --engine vllm --model "$model" --effort "$effort" \
    --comparisons "$comparisons" --gpu-memory-utilization "$util" \
    --input runs --pairs pairs.parquet --prompts prompts.json \
    --out smoke/labels.jsonl --shard 0 --shards "$slice" --reply-tokens "$tokens" 2>&1 | tee smoke/smoke.log
status=${PIPESTATUS[0]}
elapsed=$(( $(date +%s) - started ))

echo
echo "=== smoke finished in ${elapsed}s, exit $status ==="
[ "$status" -eq 0 ] || exit "$status"

"$venv/bin/python" - "$elapsed" "$slice" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

elapsed, slice_ = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in Path("smoke/labels.jsonl").read_text().splitlines() if l.strip()]
usable = [r for r in rows if r.get("verdict")]
echoed = sum(1 for r in rows if r.get("echoed"))

print(f"verdicts written : {len(rows)}")
print(f"  usable         : {len(usable)}")
print(f"  echoed         : {echoed}")
print(f"  unusable       : {len(rows) - len(usable) - echoed}")
print(f"  by comparison  : {dict(Counter(r['comparison'] for r in rows))}")
print(f"  by layer       : {dict(Counter(str(r['layer']) for r in rows))}")

if usable:
    v = usable[0]["verdict"]
    print(f"\nsample verdict fields: {sorted(v)}")
    print(f"  concept_lean={v.get('concept_lean')}  concept_expressible={v.get('concept_expressible')}")
    print(f"  reasoning: {str(v.get('reasoning',''))[:160]}")

# Startup is paid once whatever the run size, so it is separated before extrapolating; leaving the
# model load and graph capture inside a tiny smoke would make the fleet look far slower than it is.
startup = 150
work = max(1, elapsed - startup)
rate = len(rows) / work
total = len(rows) * slice_
print(f"\nthroughput: {rate:.2f} verdicts/s on one GPU (after ~{startup}s startup)")
print(f"full run is about {total:,} verdicts")
for cards in (1, 2):
    print(f"  on {cards} GPU(s): {total / max(rate * cards, 1e-9) / 3600:.1f} h + startup")
PY
