#!/bin/bash
# ============================================================================
# download_cosmos_weights.sh — Download Cosmos pretrained weights for GE-Sim
#
# Prerequisites:
#   1. Accept the license at https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
#   2. Set your HuggingFace token: export HF_TOKEN=hf_xxxxx
#      Or run: huggingface-cli login
#
# The GE-Sim transformer weights are downloaded from ModelScope (no auth needed).
# The VAE, text_encoder, tokenizer, and scheduler from HuggingFace (gated, needs auth).
#
# Usage:
#   bash scripts/download_cosmos_weights.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${SCRIPT_DIR}/cosmos_pretrained"
mkdir -p "${TARGET_DIR}"

echo "=========================================================================="
echo "[download] Target: ${TARGET_DIR}"
echo "=========================================================================="

# 1. Download GE-Sim transformer weights from ModelScope
GESIM_FILE="${TARGET_DIR}/ge_sim_cosmos_v0.1.safetensors"
if [ -f "${GESIM_FILE}" ] && [ "$(stat -c%s "${GESIM_FILE}" 2>/dev/null || stat -f%z "${GESIM_FILE}")" -gt 1000000000 ]; then
    echo "[download] GE-Sim weights already exist ($(du -h "${GESIM_FILE}" | cut -f1))"
else
    echo "[download] Downloading GE-Sim transformer weights from ModelScope..."
    wget -q --show-progress \
        "https://modelscope.cn/api/v1/models/agibot_world/Genie-Envisioner/repo?Revision=master&FilePath=ge_sim_cosmos_v0.1.safetensors" \
        -O "${GESIM_FILE}"
    echo "[download] GE-Sim weights: $(du -h "${GESIM_FILE}" | cut -f1)"
fi

# 2. Download Cosmos foundation from HuggingFace (gated repo)
echo "[download] Downloading Cosmos foundation from HuggingFace..."
echo "[download] (Requires accepted license at https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)"

python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil

target_dir = '${TARGET_DIR}'

files = [
    'vae/config.json',
    'vae/diffusion_pytorch_model.safetensors',
    'text_encoder/config.json',
    'text_encoder/model.safetensors.index.json',
    'text_encoder/model-00001-of-00002.safetensors',
    'text_encoder/model-00002-of-00002.safetensors',
    'tokenizer/special_tokens_map.json',
    'tokenizer/tokenizer_config.json',
    'tokenizer/spiece.model',
    'tokenizer/tokenizer.json',
    'scheduler/scheduler_config.json',
]

for f in files:
    subdir = os.path.dirname(f)
    os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)
    local_path = os.path.join(target_dir, f)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f'  [skip] {f} (already exists)')
        continue
    print(f'  [download] {f}...')
    downloaded = hf_hub_download(
        repo_id='nvidia/Cosmos-Predict2-2B-Video2World',
        filename=f,
    )
    shutil.copy2(downloaded, local_path)
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f'    -> {size_mb:.1f} MB')

print('[download] All Cosmos foundation files downloaded.')
"

echo "=========================================================================="
echo "[download] Done. Files:"
du -sh "${TARGET_DIR}"/*
echo "=========================================================================="
