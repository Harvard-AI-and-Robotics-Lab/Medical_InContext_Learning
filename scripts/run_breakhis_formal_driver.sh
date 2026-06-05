#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-breakhis_formal_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

export HF_HOME="${HF_HOME:-/shared/ssd_14T/home/mindazhao/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

FEATURE_GPU="${FEATURE_GPU:-0}"
QWEN_GPUS="${QWEN_GPUS:-GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57,GPU-16b40df4-c8ef-04d3-e8a5-054f40cf5ec7}"
GEMMA_GPUS="${GEMMA_GPUS:-GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb,GPU-df09dc6f-c851-343b-7a73-73f3f34b0107}"
RESNET_GPU="${RESNET_GPU:-0}"
QWEN_VLLM_DEVICES="${QWEN_VLLM_DEVICES:-0,1}"
GEMMA_VLLM_DEVICES="${GEMMA_VLLM_DEVICES:-5,6}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/shared/ssd_14T/home/mindazhao/.cache/hf_models/Qwen_Qwen3.6-27B}"
GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-/shared/ssd_14T/home/mindazhao/.cache/hf_models/google_gemma-4-31B-it}"

QWEN_SESSION="${RUN_ID}_qwen_vllm"
GEMMA_SESSION="${RUN_ID}_gemma_vllm"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/driver.log"
}

uuid_csv_to_indices() {
  local uuid_csv="$1"
  local raw
  raw="$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  local result=()
  IFS=',' read -ra wanted <<< "${uuid_csv}"
  for uuid in "${wanted[@]}"; do
    uuid="$(echo "${uuid}" | xargs)"
    local idx
    idx="$(
      printf '%s\n' "${raw}" \
        | awk -F',' -v want="${uuid}" '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($2 == want) print $1}'
    )"
    if [[ -z "${idx}" ]]; then
      echo "Unknown GPU UUID: ${uuid}" >&2
      return 1
    fi
    result+=("${idx}")
  done
  local joined
  joined="$(IFS=,; echo "${result[*]}")"
  echo "${joined}"
}

wait_for_server() {
  local port="$1"
  local name="$2"
  local url="http://127.0.0.1:${port}/v1/models"
  for _ in $(seq 1 240); do
    if curl -s "${url}" >/dev/null 2>&1; then
      log "${name} is ready on port ${port}"
      return 0
    fi
    sleep 5
  done
  log "ERROR: ${name} did not become ready on port ${port}"
  return 1
}

cleanup_servers() {
  tmux kill-session -t "${QWEN_SESSION}" 2>/dev/null || true
  tmux kill-session -t "${GEMMA_SESSION}" 2>/dev/null || true
}

log "Run id: ${RUN_ID}"
log "Logs: ${LOG_DIR}"
log "HF_HOME=${HF_HOME}"
log "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
log "Feature GPU: ${FEATURE_GPU}"
log "Qwen GPUs: ${QWEN_GPUS}"
log "Gemma GPUs: ${GEMMA_GPUS}"

log "Qwen vLLM CUDA ordinals resolved from UUIDs: ${QWEN_VLLM_DEVICES}"
log "Gemma vLLM CUDA ordinals resolved from UUIDs: ${GEMMA_VLLM_DEVICES}"

FEATURE_FILES=(
  outputs/features_clip_global_breakhis/breakhis/clip/metadata.json
  outputs/features_clip_global_breakhis/breakhis/clip/global_embeddings.npy
  outputs/features_dinov3_global_breakhis/breakhis/dinov3/metadata.json
  outputs/features_dinov3_global_breakhis/breakhis/dinov3/global_embeddings.npy
  outputs/features_clip_dinov3cls_05_global_breakhis/breakhis/clip_dinov3cls05/metadata.json
  outputs/features_clip_dinov3cls_05_global_breakhis/breakhis/clip_dinov3cls05/global_embeddings.npy
)
FEATURES_READY=1
for feature_file in "${FEATURE_FILES[@]}"; do
  if [[ ! -s "${feature_file}" ]]; then
    FEATURES_READY=0
    break
  fi
done

if [[ "${FEATURES_READY}" == "1" ]]; then
  log "Step 1/5: BreaKHis features already exist; skipping feature extraction"
else
  log "Step 1/5: extracting CLIP/DINOv3/fused BreaKHis features"
  CUDA_VISIBLE_DEVICES="${FEATURE_GPU}" bash scripts/run_breakhis_feature_pipeline.sh 2>&1 | tee "${LOG_DIR}/features.log"
fi

log "Step 2/5: starting Qwen3.6 vLLM"
tmux new-session -d -s "${QWEN_SESSION}" \
  "cd $(pwd) && source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen36_vllm && export HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' VLLM_API_KEY='${VLLM_API_KEY}' CUDA_DEVICE_ORDER='${CUDA_DEVICE_ORDER}' CUDA_VISIBLE_DEVICES='${QWEN_VLLM_DEVICES}' NCCL_P2P_DISABLE='1' NCCL_IB_DISABLE='1' MODEL_PATH='${QWEN_MODEL_PATH}' SERVED_MODEL_NAME='qwen36-27b-final' PORT='18001' TP_SIZE='2' && bash scripts/start_qwen36_vllm.sh > '${LOG_DIR}/qwen_vllm.log' 2>&1"

log "Step 2/5: starting Gemma4 vLLM"
tmux new-session -d -s "${GEMMA_SESSION}" \
  "cd $(pwd) && source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen36_vllm && export HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' VLLM_API_KEY='${VLLM_API_KEY}' CUDA_DEVICE_ORDER='${CUDA_DEVICE_ORDER}' CUDA_VISIBLE_DEVICES='${GEMMA_VLLM_DEVICES}' NCCL_P2P_DISABLE='1' NCCL_IB_DISABLE='1' MODEL_PATH='${GEMMA_MODEL_PATH}' SERVED_MODEL_NAME='gemma4-31b-final' PORT='18002' TP_SIZE='2' && bash scripts/start_gemma4_vllm.sh > '${LOG_DIR}/gemma_vllm.log' 2>&1"

wait_for_server 18001 "Qwen3.6"
wait_for_server 18002 "Gemma4"

log "Step 3/5: running Qwen3.6 BreaKHis VLM suite"
bash scripts/run_breakhis_vlm_suite.sh qwen > "${LOG_DIR}/qwen_suite.log" 2>&1 &
QWEN_PID=$!

log "Step 4/5: running Gemma4 BreaKHis VLM suite"
bash scripts/run_breakhis_vlm_suite.sh gemma > "${LOG_DIR}/gemma_suite.log" 2>&1 &
GEMMA_PID=$!

wait "${QWEN_PID}"
log "Qwen3.6 suite completed"
wait "${GEMMA_PID}"
log "Gemma4 suite completed"

cleanup_servers
log "Stopped vLLM server sessions"

log "Step 5/5: running ResNet50 supervised baseline"
CUDA_VISIBLE_DEVICES="${RESNET_GPU}" bash scripts/run_breakhis_resnet50.sh > "${LOG_DIR}/resnet50.log" 2>&1

log "All BreaKHis formal runs completed"
