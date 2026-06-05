#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${LOG_DIR:-logs/vqamed2019_shortprompt_watchdog}"
RUN_CONFIG_DIR="${RUN_CONFIG_DIR:-outputs/generated_configs/vqamed2019_shortprompt}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-shortprompt}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
FREE_USED_MB_THRESHOLD="${FREE_USED_MB_THRESHOLD:-1000}"
ALLOWED_GPUS="${ALLOWED_GPUS:-0,1,2,3,6}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-2400}"
VLLM_BIN_DIR="${VLLM_BIN_DIR:-/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin}"

mkdir -p "${LOG_DIR}" "${RUN_CONFIG_DIR}"

export PATH="${VLLM_BIN_DIR}:$PATH"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/controller.log" >&2
}

free_gpu_pair() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v threshold="${FREE_USED_MB_THRESHOLD}" -v allowed="${ALLOWED_GPUS}" '
        BEGIN {
          n = split(allowed, gpu_ids, ",");
          for (i = 1; i <= n; i++) {
            gsub(/ /, "", gpu_ids[i]);
            ok[gpu_ids[i]] = 1;
          }
        }
        {
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          if (($1 in ok) && $2 + 0 <= threshold) {
            print $1;
          }
        }' \
    | head -n 2 \
    | paste -sd, -
}

wait_for_two_free_gpus() {
  local pair
  while true; do
    pair="$(free_gpu_pair || true)"
    if [[ "${pair}" == *,* ]]; then
      echo "${pair}"
      return 0
    fi
    log "waiting for two free GPUs; current usage: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' '; ')"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

wait_for_server() {
  local name="$1"
  local base_url="$2"
  local server_pid="$3"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1; then
      log "${name} server ready at ${base_url}"
      return 0
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      log "ERROR: ${name} server process exited before readiness"
      return 1
    fi
    if (( "$(date +%s)" - start > SERVER_READY_TIMEOUT_SECONDS )); then
      log "ERROR: ${name} server not ready after ${SERVER_READY_TIMEOUT_SECONDS}s"
      return 1
    fi
    sleep 15
  done
}

make_run_config() {
  local src="$1"
  local dst="$2"
  python3 - "$src" "$dst" "$OUTPUT_SUFFIX" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
suffix = sys.argv[3]
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["output_root"] = f'{cfg["output_root"]}_{suffix}'
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
PY
}

stop_server() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

run_model_group() {
  local prefix="$1"
  local port="$2"
  local served_name="$3"
  local start_script="$4"
  local pair
  local server_pid=""
  pair="$(wait_for_two_free_gpus)"
  log "starting ${prefix} on GPUs ${pair}, port ${port}, served_name=${served_name}"

  CUDA_VISIBLE_DEVICES="${pair}" \
  VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD}" \
  PORT="${port}" \
  SERVED_MODEL_NAME="${served_name}" \
  TP_SIZE=2 \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}" \
  bash "${start_script}" > "${LOG_DIR}/${prefix}_vllm.log" 2>&1 &
  server_pid=$!

  if ! wait_for_server "${prefix}" "http://127.0.0.1:${port}/v1" "${server_pid}"; then
    stop_server "${server_pid}"
    return 1
  fi

  local configs=(
    "configs/final/vqamed2019_${prefix}_noicl_fixed6_random6.yaml"
    "configs/final/vqamed2019_${prefix}_clip_top6.yaml"
    "configs/final/vqamed2019_${prefix}_dinov3_cls_top6.yaml"
    "configs/final/vqamed2019_${prefix}_clip_dinov3cls_top6.yaml"
  )
  local cfg
  for cfg in "${configs[@]}"; do
    local run_cfg="${RUN_CONFIG_DIR}/$(basename "${cfg}")"
    make_run_config "${cfg}" "${run_cfg}"
    log "running ${prefix}: ${run_cfg}"
    python3 scripts/run_final_vqa.py --config "${run_cfg}" > "${LOG_DIR}/$(basename "${run_cfg}" .yaml).log" 2>&1
    log "finished ${prefix}: ${run_cfg}"
  done

  log "stopping ${prefix} vLLM server"
  stop_server "${server_pid}"
  log "completed ${prefix}"
}

if ! command -v vllm >/dev/null 2>&1; then
  log "ERROR: vllm not found on PATH=${PATH}"
  exit 1
fi

log "watchdog started; output_suffix=${OUTPUT_SUFFIX}; free_threshold=${FREE_USED_MB_THRESHOLD}MB; allowed_gpus=${ALLOWED_GPUS}"
run_model_group gemma4 18302 gemma4-31b-vqamed2019 scripts/start_gemma4_vllm.sh || log "FAILED: gemma4"
run_model_group medgemma27b 18303 medgemma-27b-vqamed2019 scripts/start_medgemma_vllm.sh || log "FAILED: medgemma27b"
run_model_group qwen36 18301 qwen36-27b-vqamed2019 scripts/start_qwen36_vllm.sh || log "FAILED: qwen36"
log "all VQA-Med2019 short-prompt reruns finished"
