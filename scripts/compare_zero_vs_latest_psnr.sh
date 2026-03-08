#!/bin/bash
set -euo pipefail

REPO="/apdcephfs_nj10/share_301739632/yhr/AgiBotWorld/Genie-Envisioner"
DEFAULT_RUN_DIR="${REPO}/outputs/20260304_005413_acwm_cosmos_iros2026_h20/2026_03_04_00_54_25"
ZERO_CKPT_DEFAULT="/apdcephfs_zwfy/share_303937731/yhr/Genie-Envisioner/cosmos_pretrained/ge_sim_cosmos_v0.1.safetensors"
DATA_ROOT_DEFAULT="/apdcephfs_zwfy/share_303937731/yhr/agibot/data/AgiBotWorldChallenge-2026/WorldModel/iros_challenge_2026_acwm"

RUN_DIR="${1:-${DEFAULT_RUN_DIR}}"
MAX_SAMPLES="${2:--1}"     # -1 = all validation samples
DEVICE="${3:-cuda:0}"
SPLIT="${4:-val}"          # val|validation
ZERO_CKPT="${ZERO_CKPT:-${ZERO_CKPT_DEFAULT}}"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"

if [ ! -d "${RUN_DIR}" ]; then
  echo "[compare_psnr] ERROR: run dir not found: ${RUN_DIR}"
  exit 1
fi
if [ ! -f "${ZERO_CKPT}" ]; then
  echo "[compare_psnr] ERROR: zero ckpt not found: ${ZERO_CKPT}"
  exit 1
fi

LATEST_STEP_DIR="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n 1)"
if [ -z "${LATEST_STEP_DIR}" ]; then
  echo "[compare_psnr] ERROR: no step_* directory found in ${RUN_DIR}"
  exit 1
fi
LATEST_CKPT="${LATEST_STEP_DIR}/diffusion_pytorch_model.safetensors"
if [ ! -f "${LATEST_CKPT}" ]; then
  echo "[compare_psnr] ERROR: latest ckpt not found: ${LATEST_CKPT}"
  exit 1
fi

RUN_NAME="$(basename "${RUN_DIR}")"
STEP_NAME="$(basename "${LATEST_STEP_DIR}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${REPO}/outputs/compare_zero_vs_latest/${RUN_NAME}_${STEP_NAME}_${STAMP}"
OUT_ZERO="${OUT_BASE}/zero"
OUT_LATEST="${OUT_BASE}/latest"

mkdir -p "${OUT_BASE}"

echo "========================================================================="
echo "[compare_psnr] RUN_DIR      : ${RUN_DIR}"
echo "[compare_psnr] ZERO_CKPT    : ${ZERO_CKPT}"
echo "[compare_psnr] LATEST_CKPT  : ${LATEST_CKPT}"
echo "[compare_psnr] SPLIT        : ${SPLIT}"
echo "[compare_psnr] MAX_SAMPLES  : ${MAX_SAMPLES}"
echo "[compare_psnr] DATA_ROOT    : ${DATA_ROOT}"
echo "[compare_psnr] DEVICE       : ${DEVICE}"
echo "[compare_psnr] OUT_BASE     : ${OUT_BASE}"
echo "========================================================================="

cd "${REPO}"

echo "[compare_psnr] Running ZERO ckpt inference..."
bash scripts/infer_gesim_iros2026.sh "${SPLIT}" "${ZERO_CKPT}" "${OUT_ZERO}" "${OUT_ZERO}/challenge_mp4" \
  --data_root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --max_samples "${MAX_SAMPLES}"

echo "[compare_psnr] Running LATEST ckpt inference..."
bash scripts/infer_gesim_iros2026.sh "${SPLIT}" "${LATEST_CKPT}" "${OUT_LATEST}" "${OUT_LATEST}/challenge_mp4" \
  --data_root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --max_samples "${MAX_SAMPLES}"

ZERO_METRICS="${OUT_ZERO}/metrics.json"
LATEST_METRICS="${OUT_LATEST}/metrics.json"
if [ ! -f "${ZERO_METRICS}" ] || [ ! -f "${LATEST_METRICS}" ]; then
  echo "[compare_psnr] ERROR: metrics.json missing."
  echo "  ZERO_METRICS=${ZERO_METRICS}"
  echo "  LATEST_METRICS=${LATEST_METRICS}"
  exit 1
fi

python3 - <<PY
import json
zero = json.load(open("${ZERO_METRICS}"))
latest = json.load(open("${LATEST_METRICS}"))
z_psnr = float(zero.get("avg_psnr", 0.0))
l_psnr = float(latest.get("avg_psnr", 0.0))
z_ssim = float(zero.get("avg_ssim", 0.0))
l_ssim = float(latest.get("avg_ssim", 0.0))
print("=================================================================")
print(f"ZERO   avg_psnr={z_psnr:.4f}, avg_ssim={z_ssim:.6f}, n={zero.get('n_samples')}")
print(f"LATEST avg_psnr={l_psnr:.4f}, avg_ssim={l_ssim:.6f}, n={latest.get('n_samples')}")
print(f"DELTA  psnr={l_psnr - z_psnr:+.4f}, ssim={l_ssim - z_ssim:+.6f}")
print("=================================================================")
PY

echo "[compare_psnr] Done."
