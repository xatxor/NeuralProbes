#! /usr/bin/env bash
# Prepare a rented box to judge the re-screen. Runs while the machine is still CPU-only.
#
# Everything expensive here needs no GPU: the model is a 63 GB download and vLLM ships precompiled
# CUDA kernels in its wheel. What genuinely needs the cards -- torch.compile and CUDA graph capture,
# about a hundred seconds -- happens on first model load and cannot be done in advance.
#
# Usage:  ./judgebox-setup.sh [MODEL]
set -uo pipefail

model=${1:-openai/gpt-oss-120b}
root=${ROOT:-$HOME/judge}
venv="$root/venv"

mkdir -p "$root"
cd "$root" || exit 1

echo "=== machine ==="
nproc
free -g | head -2
df -BG --output=target,avail / "$root" 2>/dev/null | tail -2
# The driver may or may not be present before the cards are attached; both are fine at this stage.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "no GPUs visible yet (expected on the CPU-only VM)"

free=$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')
echo "free disk: ${free}G"
[ "$free" -ge 90 ] || { echo "only ${free}G free; the model alone is ~63G" >&2; exit 1; }

echo "=== python environment ==="
python3 -m venv "$venv" 2>/dev/null || python3 -m venv --without-pip "$venv"
"$venv/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$venv/bin/pip" install --quiet --upgrade pip wheel

# Pinned to the version that judged the first sweep. The follow-up run measures plus and minus
# against the unsteered baseline, and those verdicts are only comparable with the 168,576 already
# collected if the judge stack is the same one. vLLM pulls its own torch, so pinning it pins most of
# the environment; the exact torch and transformers versions of the first run were never captured,
# because only the judge logs were downloaded and this printout was not among them.
#
# Nothing is built from source here -- the wheel ships precompiled CUDA kernels -- so a pin can only
# fail by rejecting the driver, which surfaces in seconds. Falling back to current is better than
# stalling the box, and the control slice in judgebox-smoke.sh is what proves comparability either way.
"$venv/bin/pip" install --quiet "vllm==0.26.0" huggingface_hub hf_transfer pyarrow \
    || { echo "!!! vllm==0.26.0 refused; falling back to current and REQUIRING the control slice" >&2
         "$venv/bin/pip" install --quiet vllm huggingface_hub hf_transfer pyarrow; }
"$venv/bin/python" - <<'PY' | tee "$root/versions.txt"
import importlib.metadata as md
for name in ("vllm", "torch", "transformers", "pyarrow"):
    try:
        print(f"{name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: MISSING")
PY
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$root/versions.txt" 2>/dev/null || true

echo "=== fetching $model ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
fetched=0
for attempt in 1 2 3 4 5; do
    if "$venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$model", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                        max_workers=8))
PY
    then
        fetched=1
        echo "weights present after attempt $attempt"
        break
    fi
    echo "download attempt $attempt failed, retrying" >&2
    sleep $((attempt * 30))
done
[ "$fetched" -eq 1 ] || { echo "$model unavailable after 5 attempts" >&2; exit 1; }

du -sh "$HOME/.cache/huggingface" 2>/dev/null
df -h "$root" | tail -1
echo "=== ready; attach the GPUs, then run judgebox-run.sh ==="
