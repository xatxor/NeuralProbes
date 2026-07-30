#! /usr/bin/env bash
# Prepare school18 (4x V100-SXM3-32GB) for the dose-response readout.
#
# V100 is sm_70: no bf16, no flash-attention. The model therefore loads in float32 and is sharded
# across the cards -- 8.19B parameters is 32.8 GB in float32, which does not fit one 32 GB card but
# fits comfortably across four. There is no generation here, only forward passes over ~50 tokens, so
# nothing about that is slow.
#
# Run detached; it pulls ~16 GB of weights.
#   setsid nohup ./dosebox-setup.sh > setup.log 2>&1 < /dev/null &
set -uo pipefail

root=${ROOT:-$HOME/dose}
venv="$root/venv"
model=${QWEN:-Qwen/Qwen3-8B}

mkdir -p "$root" && cd "$root" || exit 1

echo "=== $(date -Is) building venv ==="
python3 -m venv "$venv" || exit 1
"$venv/bin/pip" install -q --upgrade pip wheel || exit 1

# cu124 wheels still ship sm_70 kernels; cu128 and later dropped Volta.
"$venv/bin/pip" install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    || { echo "torch install FAILED" >&2; exit 1; }
"$venv/bin/pip" install -q \
    'transformers==4.57.1' 'tokenizers>=0.21' safetensors numpy pandas pyarrow scipy \
    huggingface_hub hf_transfer accelerate \
    || { echo "python deps FAILED" >&2; exit 1; }

"$venv/bin/pip" freeze > versions.txt

echo "=== $(date -Is) checking the cards see torch ==="
"$venv/bin/python" - <<'PY' || exit 1
import torch
print("torch     :", torch.__version__)
print("cuda      :", torch.version.cuda)
print("devices   :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"  gpu{i}   : {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]}")
assert torch.cuda.device_count() >= 2, "need at least two cards for float32 Qwen3-8B"
x = torch.randn(64, 64, device="cuda:0")
assert torch.isfinite(x @ x).all()
print("matmul    : ok")
PY

echo "=== $(date -Is) fetching $model ==="
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
"$venv/bin/python" - "$model" <<'PY' || { echo "model download FAILED" >&2; exit 1; }
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("cached at :", path)
PY

echo "=== $(date -Is) setup complete ==="
du -sh ~/.cache/huggingface 2>/dev/null
