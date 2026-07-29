#! /usr/bin/env bash
# Judge the re-screen on an attached pair of H200s. Run after judgebox-setup.sh and after the cards
# are visible.
#
# gpt-oss-120b is about 63 GB in MXFP4, so it fits one H200 outright with room for a large KV cache.
# Two independent single-GPU instances therefore beat splitting one model across both: no tensor
# parallelism, no NVLink dependency, and a failed card costs half the run rather than all of it.
# judge.py already shards, so each process takes every other comparison.
#
# Usage:  ./judgebox-run.sh [SHARDS] [EFFORT] [REPLY_TOKENS]
#         COMPARISONS=plus_baseline,minus_baseline ./judgebox-run.sh
#
# COMPARISONS selects which contrasts judge.py emits. The first sweep ran `steer,ablate`; this box
# exists for `plus_baseline,minus_baseline`, which is the comparison that was never made. Every
# result so far is +alpha against -alpha, and a vector wins that whether the plus arm produced the
# behaviour or the minus arm destroyed it. Only the unsteered baseline separates those.
#
# Output goes to a directory named after the comparison set, so a second run cannot overwrite the
# first's verdicts.
set -uo pipefail

shards=${1:-2}
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
out=${OUT:-verdicts-$(echo "$comparisons" | tr ',' '-')}

cd "$root" || exit 1

echo "=== cards ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
visible=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
[ "$visible" -ge "$shards" ] || { echo "asked for $shards shards but only $visible GPUs are visible" >&2; exit 1; }

rows=$(cat runs/shard-*/runs.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "generations to judge: ${rows:-0}"
[ "${rows:-0}" -gt 0 ] || { echo "no generations under runs/; upload them first" >&2; exit 1; }

echo "=== comparisons: $comparisons -> $out (gpu_memory_utilization $util) ==="
# judge.py cuts shards with items[shard::shards] and contrasts() emits the kinds in a fixed order, so
# with two kinds and two shards each card gets one kind whole: shard 0 all plus_baseline, shard 1 all
# minus_baseline. That is how the first sweep behaved too (shard 0 all steer, shard 1 all ablate). It
# is fine, but it means a dead card costs one entire comparison rather than half of each.
mkdir -p "$out"
pids=()
for shard in $(seq 0 $((shards - 1))); do
    target="$out/shard-$shard"
    mkdir -p "$target"
    echo "=== shard $shard on GPU $shard ==="
    CUDA_VISIBLE_DEVICES="$shard" HF_HUB_DISABLE_XET=1 \
        "$venv/bin/python" judge.py \
        --mode rescreen --engine vllm --model "$model" --effort "$effort" \
        --comparisons "$comparisons" --gpu-memory-utilization "$util" \
        --input runs --pairs pairs.parquet --prompts prompts.json \
        --out "$target/labels.jsonl" --shard "$shard" --shards "$shards" --reply-tokens "$tokens" \
        > "judge-$shard.log" 2>&1 &
    pids+=("$!")
    # Two processes pulling the same weights off disk at once thrashes; the second starts once the
    # first has them in page cache.
    sleep 30
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo "=== done, exit $status ==="
for shard in $(seq 0 $((shards - 1))); do
    file="$out/shard-$shard/labels.jsonl"
    echo "shard $shard: $(wc -l < "$file" 2>/dev/null | tr -d ' ') verdicts"
    tail -3 "judge-$shard.log"
done
exit "$status"
