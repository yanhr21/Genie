#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

TRAIN_VARIANT="${GESIM_TRAIN_VARIANT:-full}"
if [ "${1:-}" = "random_init" ] || [ "${1:-}" = "rand_init" ] || [ "${1:-}" = "randinit" ]; then
    TRAIN_VARIANT="random_init"
    shift
fi

DEFAULT_OFFICIAL_JEPA="pretrained_weights/vjepa2_official/vjepa2-ac-vitg.pt"

if [ "${GESIM_PDSH_WORKER:-0}" != "1" ]; then
    if [ -z "${NODE_IP_LIST:-}" ]; then
        echo "[gesim_train] ERROR: NODE_IP_LIST is not set"
        exit 1
    fi
    if ! command -v pdsh >/dev/null 2>&1; then
        echo "[gesim_train] ERROR: pdsh is not installed or not in PATH"
        exit 1
    fi

    GPU_NAME_RAW="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | tr -d '\r')"
    if [[ "${GPU_NAME_RAW}" == *"H20"* ]]; then
        if [ "${TRAIN_VARIANT}" = "random_init" ]; then
            CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_h20_random_init.yaml"
        elif [ -f "configs/cosmos_model/acwm_cosmos_iros2026_h20_train_vjepa2.yaml" ] && [ -f "${DEFAULT_OFFICIAL_JEPA}" ]; then
            CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_h20_train_vjepa2.yaml"
        else
            CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_h20.yaml"
        fi
    else
        if [ "${TRAIN_VARIANT}" = "random_init" ]; then
            CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_random_init.yaml"
        else
            CONFIG="configs/cosmos_model/acwm_cosmos_iros2026.yaml"
        fi
    fi
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
    RUN_TS="${GESIM_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

    EXTRA_ARGS_ESCAPED=""
    if [ "$#" -gt 0 ]; then
        EXTRA_ARGS_ESCAPED="$(printf ' %q' "$@")"
    fi

    REMOTE_ENV="GESIM_PDSH_WORKER=1 GESIM_RUN_TS=${RUN_TS} NODE_IP_LIST=${NODE_IP_LIST}"
    if [ -n "${GESIM_CONFIG_OVERRIDE:-}" ]; then
        REMOTE_ENV="${REMOTE_ENV} GESIM_CONFIG_OVERRIDE=${GESIM_CONFIG_OVERRIDE}"
    fi
    if [ -n "${GESIM_JEPA_CKPT:-}" ]; then
        REMOTE_ENV="${REMOTE_ENV} GESIM_JEPA_CKPT=${GESIM_JEPA_CKPT}"
    fi
    if [ -n "${TRAIN_VARIANT:-}" ]; then
        REMOTE_ENV="${REMOTE_ENV} GESIM_TRAIN_VARIANT=${TRAIN_VARIANT}"
    fi

    echo "[gesim_train] launcher config=${GESIM_CONFIG_OVERRIDE}"
    echo "[gesim_train] launcher jepa_ckpt=${GESIM_JEPA_CKPT:-}"
    echo "[gesim_train] launcher nodes=${node_ip}"

    REMOTE_CMD="cd ${SCRIPT_DIR}; ${REMOTE_ENV} bash scripts/train_gesim_iros2026.sh${EXTRA_ARGS_ESCAPED}"
    pdsh -R ssh -w "${node_ip}" -f "${worker_num}" "${REMOTE_CMD}"
    exit 0
fi

GPU_NAME_RAW="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | tr -d '\r')"
if [ -z "${GPU_NAME_RAW}" ]; then
    echo "[gesim_train] WARNING: failed to query GPU name, using default config"
    GPU_NAME_RAW="unknown"
fi

if [[ "${GPU_NAME_RAW}" == *"H20"* ]]; then
    if [ "${TRAIN_VARIANT}" = "random_init" ]; then
        CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_h20_random_init.yaml"
        GPU_CONFIG_HINT="H20 random-init config"
    elif [ -f "configs/cosmos_model/acwm_cosmos_iros2026_h20_train_vjepa2.yaml" ] && [ -f "${DEFAULT_OFFICIAL_JEPA}" ]; then
        CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_h20_train_vjepa2.yaml"
        GPU_CONFIG_HINT="H20 vjepa2 train config"
    else
        CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_h20.yaml"
        GPU_CONFIG_HINT="H20 full config"
    fi
else
    if [ "${TRAIN_VARIANT}" = "random_init" ]; then
        CONFIG="configs/cosmos_model/acwm_cosmos_iros2026_random_init.yaml"
        GPU_CONFIG_HINT="default random-init config"
    else
        CONFIG="configs/cosmos_model/acwm_cosmos_iros2026.yaml"
        GPU_CONFIG_HINT="default full config"
    fi
fi
if [ -n "${GESIM_CONFIG_OVERRIDE:-}" ]; then
    CONFIG="${GESIM_CONFIG_OVERRIDE}"
    GPU_CONFIG_HINT="override config"
fi
CONFIG_BASENAME="$(basename "${CONFIG}")"
CONFIG_STEM="${CONFIG_BASENAME%.*}"
RUN_TS="${GESIM_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_TS}_${CONFIG_STEM}"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

RUNNER_CLASS_PATH="runner/ge_trainer.py"
RUNNER_CLASS="Trainer"

if command -v python3 >/dev/null 2>&1; then
    CONFIG_LOAD_WEIGHTS="$(python3 - <<PY
import yaml
with open("${CONFIG}", "r") as f:
    cfg = yaml.safe_load(f)
print(str(cfg.get("load_weights", "UNKNOWN")))
PY
)"
else
    CONFIG_LOAD_WEIGHTS="UNKNOWN"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_TIMEOUT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

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
        echo "[gesim_train] ERROR: local IP not in NODE_IP_LIST"
        exit 1
    fi
fi

MASTER_PORT="${MASTER_PORT:-29507}"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

mkdir -p "${RUN_OUTPUT_DIR}"
LOG_FILE="${RUN_OUTPUT_DIR}/train_node${NODE_RANK}.log"
if [ "${NODE_RANK}" -eq 0 ]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
else
    exec >> "${LOG_FILE}" 2>&1
fi

ENABLE_OCCUPY_ON_EXIT="${ENABLE_OCCUPY_ON_EXIT:-1}"
OCCUPY_FALLBACK_SCRIPT="${OCCUPY_FALLBACK_SCRIPT:-/apdcephfs_nj8/share_301739632/yestinlong/tools/debug/run_pdsh_pretrain_l.sh}"
cleanup_on_exit() {
    local exit_code=$?
    trap - EXIT
    if [ "${ENABLE_OCCUPY_ON_EXIT}" = "1" ] && [ "${NODE_RANK}" -eq 0 ]; then
        echo "[gesim_train] EXIT code=${exit_code}, run occupy fallback: bash ${OCCUPY_FALLBACK_SCRIPT}"
        if [ -f "${OCCUPY_FALLBACK_SCRIPT}" ]; then
            set +e
            bash "${OCCUPY_FALLBACK_SCRIPT}"
            local occupy_rc=$?
            set -e
            if [ "${occupy_rc}" -ne 0 ]; then
                echo "[gesim_train] WARNING: occupy fallback failed with code ${occupy_rc}"
            fi
        else
            echo "[gesim_train] WARNING: occupy fallback script not found: ${OCCUPY_FALLBACK_SCRIPT}"
        fi
    fi
    exit "${exit_code}"
}
trap cleanup_on_exit EXIT

echo "=========================================================================="
echo "[gesim_train] ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU(s) = ${TOTAL_GPUS} GPUs"
echo "[gesim_train] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}, NODE_RANK=${NODE_RANK}"
echo "[gesim_train] GPU: ${GPU_NAME_RAW} -> ${GPU_CONFIG_HINT}"
echo "[gesim_train] Variant: ${TRAIN_VARIANT}"
echo "[gesim_train] Config: ${CONFIG}"
echo "[gesim_train] Config load_weights: ${CONFIG_LOAD_WEIGHTS}"
echo "[gesim_train] JEPA ckpt: ${GESIM_JEPA_CKPT:-}"
echo "[gesim_train] Run output dir: ${RUN_OUTPUT_DIR}"
echo "[gesim_train] Terminal log file: ${LOG_FILE}"
echo "=========================================================================="

torchrun \
    --nnodes ${NUM_NODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${NODE_RANK} \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id "gesim_iros2026" \
    main.py \
    --config_file ${CONFIG} \
    --mode train \
    --output_path ${RUN_OUTPUT_DIR} \
    --runner_class_path ${RUNNER_CLASS_PATH} \
    --runner_class ${RUNNER_CLASS} \
    "$@"
