#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/pretrained_weights"
V2_DIR="${OUT_ROOT}/vjepa2_official"
V1_DIR="${OUT_ROOT}/vjepa_official"
mkdir -p "${V2_DIR}/evals" "${V1_DIR}"

retry_cmd() {
  local attempts="$1"
  local sleep_secs="$2"
  shift 2
  local i
  for ((i=1; i<=attempts; i++)); do
    if "$@"; then
      return 0
    fi
    if (( i < attempts )); then
      echo "[retry] attempt=${i}/${attempts} cmd=$*" >&2
      sleep "${sleep_secs}"
    fi
  done
  return 1
}

remote_size() {
  local url="$1"
  python3 - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
req = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(req, timeout=60) as resp:
    print(resp.headers.get("Content-Length", ""))
PY
}

download_one() {
  local url="$1"
  local out="$2"
  local exp
  local cur=0
  exp="$(retry_cmd 5 10 remote_size "$url")"
  if [[ -f "$out" ]]; then
    cur="$(stat -c '%s' "$out" 2>/dev/null || echo 0)"
  fi
  if [[ -n "$exp" && "$cur" == "$exp" ]]; then
    echo "[skip-ok] $(basename "$out") size=${cur}"
    return 0
  fi
  echo "[download] $(basename "$out") current=${cur} expected=${exp}"
  retry_cmd 5 15 wget -c -nv -O "$out" "$url"
  cur="$(stat -c '%s' "$out" 2>/dev/null || echo 0)"
  if [[ -n "$exp" && "$cur" != "$exp" ]]; then
    echo "[size-mismatch] $(basename "$out") current=${cur} expected=${exp}" >&2
    return 1
  fi
  echo "[done] $(basename "$out") size=${cur}"
}

run_queue() {
  local base_dir="$1"
  while IFS=' ' read -r url rel; do
    [[ -n "${url}" ]] || continue
    local out="${base_dir}/${rel}"
    mkdir -p "$(dirname "$out")"
    download_one "$url" "$out"
  done
}

run_queue "${V2_DIR}" <<'EOF'
https://dl.fbaipublicfiles.com/vjepa2/vitl.pt vitl.pt
https://dl.fbaipublicfiles.com/vjepa2/vith.pt vith.pt
https://dl.fbaipublicfiles.com/vjepa2/vitg.pt vitg.pt
https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt vitg-384.pt
https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt vjepa2-ac-vitg.pt
https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt evals/ssv2-vitl-16x2x3.pt
https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt evals/ssv2-vitg-384-64x2x3.pt
https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitl-256.pt evals/diving48-vitl-256.pt
https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitg-384-32x4x3.pt evals/diving48-vitg-384-32x4x3.pt
https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitl-256.pt evals/ek100-vitl-256.pt
https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt evals/ek100-vitg-384.pt
EOF

run_queue "${V1_DIR}" <<'EOF'
https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar vitl16.pth.tar
https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar vith16.pth.tar
https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar vith16-384.pth.tar
EOF

echo "[all-done] V-JEPA/V-JEPA2 official checkpoints are complete."
