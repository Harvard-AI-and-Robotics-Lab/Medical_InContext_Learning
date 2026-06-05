#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DELAY_SECONDS="${DELAY_SECONDS:-7200}"
GEMMA_GPUS="${GEMMA_GPUS:-0,1}"
MEDGEMMA_GPUS="${MEDGEMMA_GPUS:-2,3}"
LOG_DIR="${LOG_DIR:-logs/vqa_rad_delayed_remaining}"
mkdir -p "${LOG_DIR}"

VLLM_BIN_DIR="${VLLM_BIN_DIR:-/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin}"
export PATH="${VLLM_BIN_DIR}:$PATH"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/controller.log"
}

wait_for_server() {
  local name="$1"
  local url="$2"
  local deadline_seconds="${3:-1800}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "${url}/models" >/dev/null 2>&1; then
      log "${name} server is ready at ${url}"
      return 0
    fi
    if (( "$(date +%s)" - start > deadline_seconds )); then
      log "ERROR: ${name} server did not become ready within ${deadline_seconds}s"
      return 1
    fi
    sleep 15
  done
}

cleanup() {
  local code=$?
  log "cleanup: stopping vLLM servers if still running"
  if [[ -n "${GEMMA_PID:-}" ]] && kill -0 "${GEMMA_PID}" >/dev/null 2>&1; then
    kill "${GEMMA_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MEDGEMMA_PID:-}" ]] && kill -0 "${MEDGEMMA_PID}" >/dev/null 2>&1; then
    kill "${MEDGEMMA_PID}" >/dev/null 2>&1 || true
  fi
  wait "${GEMMA_PID:-}" "${MEDGEMMA_PID:-}" >/dev/null 2>&1 || true
  log "cleanup complete; exit_code=${code}"
  exit "${code}"
}
trap cleanup EXIT INT TERM

log "scheduled VQA-RAD remaining LLM run; sleeping ${DELAY_SECONDS}s"
sleep "${DELAY_SECONDS}"

log "starting Gemma4 vLLM on GPUs ${GEMMA_GPUS}"
CUDA_VISIBLE_DEVICES="${GEMMA_GPUS}" \
PORT=18402 \
SERVED_MODEL_NAME=gemma4-31b-vqa-rad \
TP_SIZE=2 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS="${GEMMA_MAX_NUM_SEQS:-8}" \
MAX_NUM_BATCHED_TOKENS="${GEMMA_MAX_NUM_BATCHED_TOKENS:-8192}" \
GPU_MEMORY_UTILIZATION="${GEMMA_GPU_MEMORY_UTILIZATION:-0.85}" \
bash scripts/start_gemma4_vllm.sh > "${LOG_DIR}/gemma4_vllm.log" 2>&1 &
GEMMA_PID=$!

log "starting MedGemma vLLM on GPUs ${MEDGEMMA_GPUS}"
CUDA_VISIBLE_DEVICES="${MEDGEMMA_GPUS}" \
PORT=18403 \
SERVED_MODEL_NAME=medgemma-27b-vqa-rad \
TP_SIZE=2 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS="${MEDGEMMA_MAX_NUM_SEQS:-8}" \
MAX_NUM_BATCHED_TOKENS="${MEDGEMMA_MAX_NUM_BATCHED_TOKENS:-8192}" \
GPU_MEMORY_UTILIZATION="${MEDGEMMA_GPU_MEMORY_UTILIZATION:-0.85}" \
bash scripts/start_medgemma_vllm.sh > "${LOG_DIR}/medgemma27b_vllm.log" 2>&1 &
MEDGEMMA_PID=$!

wait_for_server "Gemma4" "http://127.0.0.1:18402/v1"
wait_for_server "MedGemma" "http://127.0.0.1:18403/v1"

log "starting Gemma4 VQA-RAD 6-experiment suite"
bash scripts/run_vqa_rad_vqa_suite.sh gemma4 > "${LOG_DIR}/gemma4_eval.log" 2>&1 &
GEMMA_EVAL_PID=$!

log "starting MedGemma VQA-RAD 6-experiment suite"
bash scripts/run_vqa_rad_vqa_suite.sh medgemma27b > "${LOG_DIR}/medgemma27b_eval.log" 2>&1 &
MEDGEMMA_EVAL_PID=$!

wait "${GEMMA_EVAL_PID}"
log "Gemma4 VQA-RAD suite finished"
wait "${MEDGEMMA_EVAL_PID}"
log "MedGemma VQA-RAD suite finished"
