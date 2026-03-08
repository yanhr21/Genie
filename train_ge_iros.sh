#!/bin/bash
# Training script for Genie-Envisioner on IROS Challenge data
# Usage: bash train_ge_iros.sh [NUM_GPUS]

set -e

NUM_GPUS=${1:-1}
CONFIG="configs/ltx_model/iros_challenge_video.yaml"

cd /apdcephfs_nj10/share_301739632/yhr/agibot/Genie-Envisioner

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

echo "Starting GE-Base training on IROS Challenge data"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    main.py \
    --config_file $CONFIG \
    --mode train
