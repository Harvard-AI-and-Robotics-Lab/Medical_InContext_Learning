#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slake_vqa"
mkdir -p "${LOG_DIR}"

VLLM_BIN_DIR="${VLLM_BIN_DIR:-/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin}"
export PATH="${VLLM_BIN_DIR}:$PATH"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

QWEN_GPUS="${QWEN_GPUS:-2,3}"
GEMMA_GPUS="${GEMMA_GPUS:-4,5}"
MEDGEMMA_GPUS="${MEDGEMMA_GPUS:-6,7}"

default_model_path() {
  local env_value="$1"
  local local_path="$2"
  local hf_id="$3"
  if [[ -n "${env_value}" ]]; then
    echo "${env_value}"
  elif [[ -d "${local_path}" ]]; then
    echo "${local_path}"
  else
    echo "${hf_id}"
  fi
}

QWEN_MODEL_PATH="$(default_model_path "${QWEN_MODEL_PATH:-}" "/data/home/mindazhao/hf_models/Qwen_Qwen3.6-27B" "Qwen/Qwen3.6-27B")"
GEMMA_MODEL_PATH="$(default_model_path "${GEMMA_MODEL_PATH:-}" "/data/home/mindazhao/hf_models/google_gemma-4-31B-it" "google/gemma-4-31B-it")"
MEDGEMMA_MODEL_PATH="$(default_model_path "${MEDGEMMA_MODEL_PATH:-}" "/data/home/mindazhao/hf_models/google_medgemma-27b-it" "google/medgemma-27b-it")"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

start_session() {
  local session="$1"
  local cmd="$2"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "${session} already exists"
    return
  fi
  tmux new-session -d -s "${session}" "bash -lc '${cmd}'"
  echo "started ${session}"
}

start_session slake_qwen36_vllm \
  "export PATH='${VLLM_BIN_DIR}':\$PATH; cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${QWEN_GPUS}' MODEL_PATH='${QWEN_MODEL_PATH}' SERVED_MODEL_NAME='qwen36-27b-slake' PORT='18101' TP_SIZE='2' GPU_MEMORY_UTILIZATION='0.90' MAX_MODEL_LEN='${MAX_MODEL_LEN}' MAX_NUM_SEQS='${MAX_NUM_SEQS}' MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}' GENERATION_CONFIG='${GENERATION_CONFIG}' bash scripts/start_qwen36_vllm.sh > '${LOG_DIR}/qwen36_vllm.log' 2>&1"

start_session slake_gemma4_vllm \
  "export PATH='${VLLM_BIN_DIR}':\$PATH; cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${GEMMA_GPUS}' MODEL_PATH='${GEMMA_MODEL_PATH}' SERVED_MODEL_NAME='gemma4-31b-slake' PORT='18102' TP_SIZE='2' GPU_MEMORY_UTILIZATION='0.90' MAX_MODEL_LEN='${MAX_MODEL_LEN}' MAX_NUM_SEQS='${MAX_NUM_SEQS}' MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}' GENERATION_CONFIG='${GENERATION_CONFIG}' bash scripts/start_gemma4_vllm.sh > '${LOG_DIR}/gemma4_vllm.log' 2>&1"

start_session slake_medgemma27b_vllm \
  "export PATH='${VLLM_BIN_DIR}':\$PATH; cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${MEDGEMMA_GPUS}' MODEL_PATH='${MEDGEMMA_MODEL_PATH}' SERVED_MODEL_NAME='medgemma-27b-slake' PORT='18103' TP_SIZE='2' GPU_MEMORY_UTILIZATION='0.90' MAX_MODEL_LEN='${MAX_MODEL_LEN}' MAX_NUM_SEQS='${MAX_NUM_SEQS}' MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}' GENERATION_CONFIG='${GENERATION_CONFIG}' bash scripts/start_medgemma_vllm.sh > '${LOG_DIR}/medgemma27b_vllm.log' 2>&1"

echo "logs: ${LOG_DIR}"
echo "ports: qwen=18101 gemma=18102 medgemma=18103"
