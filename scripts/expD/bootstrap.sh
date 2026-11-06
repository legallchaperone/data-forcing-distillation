#!/bin/bash
# One-shot environment bootstrap for the Exp D sandbox (molab, RTX Pro 6000, uv available).
# Usage: bash bootstrap.sh [branch]   -> clones the repo to /tmp/fg/src, creates /tmp/fg/venv, downloads Wan2.1-1.3B.
set -o pipefail
BRANCH=${1:-claude/quirky-hypatia-4l9m82}
mkdir -p /tmp/fg && cd /tmp/fg
echo "== CLONE $(date -u)"
[ -d src ] || git clone -q --depth 1 -b "$BRANCH" https://github.com/legallchaperone/data-forcing-distillation src
(cd src && git pull -q && git log --oneline -1)
echo "== VENV $(date -u)"
[ -d /tmp/fg/venv ] || uv venv -q --system-site-packages /tmp/fg/venv
grep -v -E "^numpy|^torch|^jupyter" src/requirements.txt > /tmp/fg/req.txt
uv pip install -q --python /tmp/fg/venv/bin/python -r /tmp/fg/req.txt "setuptools<80" "pandas<3" "wandb<0.21" "huggingface_hub[cli]" 2>&1 | tail -1
uv pip install -q --python /tmp/fg/venv/bin/python --no-deps -e src 2>&1 | tail -1
cat > /tmp/fg/env.sh <<'EOF'
export FASTGEN_OUTPUT_ROOT=/tmp/fg/out DATA_ROOT_DIR=/tmp/fg/out/DATA CKPT_ROOT_DIR=/tmp/fg/out/MODEL HF_HOME=/tmp/fg/hf PYTHONPATH=/tmp/fg/src PYTHONUNBUFFERED=1
PY=/tmp/fg/venv/bin/python
EOF
source /tmp/fg/env.sh
$PY -c "import fastgen.methods, diffusers, webdataset, omegaconf, loguru, timm, wandb; print('DEPS_OK', wandb.__version__)" || exit 1
echo "== HF DOWNLOAD $(date -u)"
mkdir -p $HF_HOME /tmp/fg/out/DATA
/tmp/fg/venv/bin/huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --exclude "*.msgpack" "*.h5" 2>&1 | tail -1
du -sh $HF_HOME
echo "== BOOTSTRAP_DONE $(date -u)"
