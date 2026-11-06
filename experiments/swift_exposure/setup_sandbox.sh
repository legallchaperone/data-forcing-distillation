#!/usr/bin/env bash
# One-shot setup for a fresh molab GPU sandbox (RTX PRO 6000 / Blackwell, torch 2.11 cu130 preinstalled).
# Usage:  bash setup_sandbox.sh [WORK_DIR] [PYTHON]
#   WORK_DIR default /marimo/work ; PYTHON default /tmp/uv-venv/bin/python (the marimo kernel's venv)
# Steps: clone Self-Forcing, install deps (pin huggingface-hub<2 so transformers imports), apply compatibility
# patches, download Wan2.1-T2V-1.3B (~17 GB) and the Self-Forcing ode_init.pt (~5.6 GB). Idempotent.
set -euo pipefail
WORK=${1:-/marimo/work}
PY=${2:-/tmp/uv-venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK" && cd "$WORK"

[ -d sf ] || git clone --depth 1 https://github.com/guandeh17/Self-Forcing.git sf

uv pip install --python "$PY" diffusers omegaconf easydict ftfy imageio-ffmpeg av accelerate \
  opencv-python-headless sentencepiece "huggingface-hub>=1.5,<2" 2>&1 | tail -3

"$PY" "$HERE/patch_selfforcing.py" sf

if [ ! -f sf/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors ]; then
  hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir sf/wan_models/Wan2.1-T2V-1.3B \
    --exclude "assets/*" --exclude "examples/*" 2>&1 | tail -2
fi
if [ ! -f sf/checkpoints/ode_init.pt ]; then
  hf download gdhe17/Self-Forcing checkpoints/ode_init.pt vidprom_filtered_extended.txt --local-dir sf 2>&1 | tail -2
fi

cd sf && "$PY" -c "import pipeline; import wan.modules.attention as a; print('import ok; flash-attn available:', a.FLASH_ATTN_2_AVAILABLE)"
du -sh "$WORK/sf/wan_models" "$WORK/sf/checkpoints"
echo SETUP_DONE
