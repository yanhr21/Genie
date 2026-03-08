#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_OFFICIAL_JEPA="pretrained_weights/vjepa2_official/vjepa2-ac-vitg.pt"

choose_config() {
    local gpu_name_raw="$1"
    if [[ "${gpu_name_raw}" == *"H20"* ]]; then
        if [ -f "configs/cosmos_model/acwm_cosmos_iros2026_h20_val_vjepa2.yaml" ] && [ -f "${DEFAULT_OFFICIAL_JEPA}" ]; then
            echo "configs/cosmos_model/acwm_cosmos_iros2026_h20_val_vjepa2.yaml"
        else
            echo "configs/cosmos_model/acwm_cosmos_iros2026_h20_val.yaml"
        fi
    else
        echo "configs/cosmos_model/acwm_cosmos_iros2026.yaml"
    fi
}

if [ "${GESIM_PDSH_VAL_WORKER:-0}" != "1" ]; then
    if [ -z "${NODE_IP_LIST:-}" ]; then
        echo "[gesim_val] ERROR: NODE_IP_LIST is not set"
        exit 1
    fi
    if ! command -v pdsh >/dev/null 2>&1; then
        echo "[gesim_val] ERROR: pdsh is not installed or not in PATH"
        exit 1
    fi

    GPU_NAME_RAW="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | tr -d '\r')"
    CONFIG="$(choose_config "${GPU_NAME_RAW}")"
    if [ -n "${GESIM_CONFIG_OVERRIDE:-}" ]; then
        CONFIG="${GESIM_CONFIG_OVERRIDE}"
    fi
    export GESIM_CONFIG_OVERRIDE="${CONFIG}"

    if [ -z "${GESIM_JEPA_CKPT:-}" ] && [ -f "${DEFAULT_OFFICIAL_JEPA}" ] && [[ "${CONFIG}" == *"vjepa2"* ]]; then
        export GESIM_JEPA_CKPT="${DEFAULT_OFFICIAL_JEPA}"
    fi

    export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
    IFS=',' read -ra NODE_ARRAY <<< "$node_ip"
    worker_num=${#NODE_ARRAY[@]}

    EXTRA_ARGS_ESCAPED=""
    if [ "$#" -gt 0 ]; then
        EXTRA_ARGS_ESCAPED="$(printf ' %q' "$@")"
    fi

    REMOTE_ENV="GESIM_PDSH_VAL_WORKER=1 NODE_IP_LIST=${NODE_IP_LIST} GESIM_CONFIG_OVERRIDE=${GESIM_CONFIG_OVERRIDE}"
    if [ -n "${GESIM_JEPA_CKPT:-}" ]; then
        REMOTE_ENV="${REMOTE_ENV} GESIM_JEPA_CKPT=${GESIM_JEPA_CKPT}"
    fi

    echo "[gesim_val] launcher config=${GESIM_CONFIG_OVERRIDE}"
    echo "[gesim_val] launcher jepa_ckpt=${GESIM_JEPA_CKPT:-}"
    echo "[gesim_val] launcher nodes=${node_ip}"

    REMOTE_CMD="cd ${SCRIPT_DIR}; ${REMOTE_ENV} bash scripts/val_gesim_iros2026.sh${EXTRA_ARGS_ESCAPED}"
    pdsh -R ssh -w "${node_ip}" -f "${worker_num}" "${REMOTE_CMD}"
    exit 0
fi

GPU_NAME_RAW="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | tr -d '\r')"
if [ -z "${GPU_NAME_RAW}" ]; then
    GPU_NAME_RAW="unknown"
fi
CONFIG="$(choose_config "${GPU_NAME_RAW}")"
if [ -n "${GESIM_CONFIG_OVERRIDE:-}" ]; then
    CONFIG="${GESIM_CONFIG_OVERRIDE}"
fi

SPLIT="${1:-validation}"
CHECKPOINT="${2:-}"
OUTPUT_DIR="${3:-outputs/gesim_iros2026_infer/${SPLIT}}"
VIDEO_OUTPUT_ROOT="${4:-${OUTPUT_DIR}/challenge_mp4}"

NSHIFT=0
[ $# -ge 1 ] && NSHIFT=1
[ $# -ge 2 ] && NSHIFT=2
[ $# -ge 3 ] && NSHIFT=3
[ $# -ge 4 ] && NSHIFT=4
shift ${NSHIFT}

if [ "${SPLIT}" = "val" ]; then
    SPLIT="validation"
fi
DATA_ROOT="/apdcephfs_nj10/share_301739632/yhr/AgiBotWorld/data/AgiBotWorldChallenge-2026/WorldModel/iros_challenge_2026_acwm"

export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
IFS=',' read -ra NODE_ARRAY <<< "$node_ip"
NUM_NODES=${#NODE_ARRAY[@]}
MASTER_ADDR="${NODE_ARRAY[0]}"
GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)

if [ -n "${INDEX:-}" ]; then
    NODE_RANK=${INDEX}
else
    LOCAL_IPS=$(hostname -I 2>/dev/null || echo "")
    NODE_RANK=-1
    for i in "${!NODE_ARRAY[@]}"; do
        for lip in $LOCAL_IPS; do
            if [ "$lip" = "${NODE_ARRAY[$i]}" ]; then
                NODE_RANK=$i
                break 2
            fi
        done
    done
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "[gesim_val] ERROR: local IP not in NODE_IP_LIST"
        exit 1
    fi
fi

MASTER_PORT="${MASTER_PORT:-29517}"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/val_node${NODE_RANK}.log"
if [ "${NODE_RANK}" -eq 0 ]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
else
    exec >> "${LOG_FILE}" 2>&1
fi

EXTRA_ARGS=()
if [ -n "${CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${GESIM_JEPA_CKPT:-}" ]; then
    EXTRA_ARGS+=(--jepa_ckpt "${GESIM_JEPA_CKPT}")
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "=========================================================================="
echo "[gesim_val] ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU(s) = ${TOTAL_GPUS} GPUs"
echo "[gesim_val] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}, NODE_RANK=${NODE_RANK}"
echo "[gesim_val] GPU: ${GPU_NAME_RAW}"
echo "[gesim_val] Config: ${CONFIG}"
echo "[gesim_val] Split: ${SPLIT}"
echo "[gesim_val] Checkpoint: ${CHECKPOINT}"
echo "[gesim_val] JEPA ckpt: ${GESIM_JEPA_CKPT:-}"
echo "[gesim_val] Output: ${OUTPUT_DIR}"
echo "[gesim_val] MP4 Root: ${VIDEO_OUTPUT_ROOT}"
echo "=========================================================================="

torchrun \
    --nnodes ${NUM_NODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${NODE_RANK} \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id "gesim_iros2026_val" \
    scripts/infer_iros2026_gesim.py \
    --config_file "${CONFIG}" \
    --split "${SPLIT}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --video_output_root "${VIDEO_OUTPUT_ROOT}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
