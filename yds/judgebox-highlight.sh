#! /usr/bin/env bash
# Read every concept off every token of the twenty-five demonstrations, before the judge starts.
#
# Ordering is not a preference. The judge runs at gpu_memory_utilization 0.97, which leaves nothing
# beside it on the card; Qwen3-8B truncated at block 25 needs ~11 GiB. So this runs on a bare box and
# exits before judgebox-run.sh is launched. It costs about three minutes, nearly all of it the 16 GB
# model pull and load -- the arithmetic itself is 12,834 tokens across 75 sequences.
#
# Usage:  ./judgebox-highlight.sh
set -uo pipefail

root=${ROOT:-$HOME/judge}
venv="$root/venv"
model=${QWEN:-Qwen/Qwen3-8B}
out=${OUT:-top25-readout.safetensors}

cd "$root" || exit 1

echo "=== cards ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l | tr -d ' ')
[ "${busy:-0}" -eq 0 ] || {
    echo "something is already on the GPUs; this needs a bare card. Aborting." >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
}

for f in highlight.py top25.json diff.safetensors; do
    [ -s "$f" ] || { echo "missing $f; run judgebox-upload.sh first" >&2; exit 1; }
done

echo "=== fetching $model ==="
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
"$venv/bin/huggingface-cli" download "$model" >/dev/null 2>&1 \
    || "$venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('$model')
" || { echo "model download failed" >&2; exit 1; }

echo "=== reading out ==="
started=$(date +%s)
CUDA_VISIBLE_DEVICES=0 "$venv/bin/python" highlight.py \
    --source top25.json --vectors diff.safetensors --model "$model" --out "$out" \
    2>&1 | tee highlight.log
status=${PIPESTATUS[0]}
echo "=== finished in $(( $(date +%s) - started ))s, exit $status ==="
[ "$status" -eq 0 ] || exit "$status"

# Verify by reading the file back rather than trusting the exit code: a run that writes nothing still
# exits 0, which is how 245 of 248 smoke verdicts were lost once.
"$venv/bin/python" - "$out" <<'PY'
import json, struct, sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as handle:
    size = struct.unpack("<Q", handle.read(8))[0]
    header = json.loads(handle.read(size))
manifest = json.loads(header["__metadata__"]["manifest"])
keys = [k for k in header if k != "__metadata__"]

print(f"file      : {path} ({path.stat().st_size/1e6:.1f} MB)")
print(f"sequences : {len(keys)} (expected 75)")
print(f"layers    : {manifest['layers']}")
print(f"axes      : {manifest['axes']}")
arms = {}
for row in manifest["index"]:
    arms[row["arm"]] = arms.get(row["arm"], 0) + 1
print(f"by arm    : {arms}")
tokens = sum(row["shape"][0] for row in manifest["index"])
print(f"tokens    : {tokens:,}")
if len(keys) != 75:
    sys.exit(f"FAIL: expected 75 sequences, got {len(keys)}")
if tokens < 5000:
    sys.exit(f"FAIL: only {tokens} tokens scored; something truncated")
print("VERIFIED")
PY
[ $? -eq 0 ] || { echo "=== readout verification FAILED ===" >&2; exit 1; }

echo
echo "Now free the card before judging:  nvidia-smi  (should show no compute apps)"
echo "Then:  COMPARISONS=plus_baseline,minus_baseline ./judgebox-smoke.sh 500"
