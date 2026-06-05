#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-remote_breakhis_binary_ddr_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export HF_HOME="${HF_HOME:-$HOME/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3.6-27B}"
GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:?Set GEMMA_MODEL_PATH to local Gemma4-31B-it path or Hugging Face ID.}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:?Set QWEN_CUDA_VISIBLE_DEVICES to two GPU UUIDs, comma-separated.}"
GEMMA_CUDA_VISIBLE_DEVICES="${GEMMA_CUDA_VISIBLE_DEVICES:?Set GEMMA_CUDA_VISIBLE_DEVICES to two GPU UUIDs, comma-separated.}"
FEATURE_CUDA_VISIBLE_DEVICES="${FEATURE_CUDA_VISIBLE_DEVICES:-}"
RESNET_CUDA_VISIBLE_DEVICES="${RESNET_CUDA_VISIBLE_DEVICES:-}"
KEEP_VLLM="${KEEP_VLLM:-1}"

QWEN_SESSION="${RUN_ID}_qwen_vllm"
GEMMA_SESSION="${RUN_ID}_gemma_vllm"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/driver.log"
}

wait_for_server() {
  local port="$1"
  local name="$2"
  local url="http://127.0.0.1:${port}/v1/models"
  for _ in $(seq 1 240); do
    if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "${url}" >/dev/null 2>&1; then
      log "${name} ready on port ${port}"
      return 0
    fi
    sleep 5
  done
  log "ERROR: ${name} did not become ready on port ${port}"
  return 1
}

server_ready() {
  local port="$1"
  curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

log "run_id=${RUN_ID}"
log "log_dir=${LOG_DIR}"
log "HF_HOME=${HF_HOME}"
log "Qwen GPUs=${QWEN_CUDA_VISIBLE_DEVICES}"
log "Gemma GPUs=${GEMMA_CUDA_VISIBLE_DEVICES}"
log "Feature GPU(s)=${FEATURE_CUDA_VISIBLE_DEVICES:-unset}"
log "ResNet GPU(s)=${RESNET_CUDA_VISIBLE_DEVICES:-unset}"

log "Step 1: prepare BreaKHis binary manifest from patient-level split"
python scripts/prepare_breakhis_binary_manifest.py > "${LOG_DIR}/prepare_breakhis_binary.log" 2>&1

log "Step 2: preprocess DDR images to crop/pad 1024 if needed"
python scripts/preprocess_ddr_images.py \
  --manifest-csv manifests/ddr_official_split.csv \
  --data-root . \
  --output-root data/processed/ddr_crop_pad_1024 \
  --output-manifest manifests/ddr_official_split_crop_pad_1024.csv \
  --summary-json manifests/ddr_official_split_crop_pad_1024.summary.json \
  --size 1024 \
  --threshold 10 \
  --margin 8 \
  --quality 95 \
  --workers "${DDR_PREPROCESS_WORKERS:-16}" \
  --skip-existing > "${LOG_DIR}/preprocess_ddr_1024.log" 2>&1

log "Step 3: extract BreaKHis binary CLIP/DINOv3/fused features"
if [[ -n "${FEATURE_CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${FEATURE_CUDA_VISIBLE_DEVICES}" bash scripts/run_breakhis_binary_feature_pipeline.sh > "${LOG_DIR}/breakhis_binary_features.log" 2>&1
else
  bash scripts/run_breakhis_binary_feature_pipeline.sh > "${LOG_DIR}/breakhis_binary_features.log" 2>&1
fi

if server_ready 18001; then
  log "Qwen3.6 vLLM already running on port 18001"
else
  log "Step 4: start Qwen3.6 vLLM"
  tmux new-session -d -s "${QWEN_SESSION}" \
    "cd $(pwd) && source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen36_vllm && export HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' VLLM_API_KEY='${VLLM_API_KEY}' CUDA_DEVICE_ORDER='${CUDA_DEVICE_ORDER}' CUDA_VISIBLE_DEVICES='${QWEN_CUDA_VISIBLE_DEVICES}' MODEL_PATH='${QWEN_MODEL_PATH}' SERVED_MODEL_NAME='qwen36-27b-final' PORT='18001' TP_SIZE='2' GPU_MEMORY_UTILIZATION='0.90' MAX_MODEL_LEN='32768' MAX_NUM_SEQS='16' NCCL_P2P_DISABLE='1' NCCL_IB_DISABLE='1' && bash scripts/start_qwen36_vllm.sh > '${LOG_DIR}/qwen_vllm.log' 2>&1"
fi

if server_ready 18002; then
  log "Gemma4 vLLM already running on port 18002"
else
  log "Step 4: start Gemma4 vLLM"
  tmux new-session -d -s "${GEMMA_SESSION}" \
    "cd $(pwd) && source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen36_vllm && export HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' VLLM_API_KEY='${VLLM_API_KEY}' CUDA_DEVICE_ORDER='${CUDA_DEVICE_ORDER}' CUDA_VISIBLE_DEVICES='${GEMMA_CUDA_VISIBLE_DEVICES}' MODEL_PATH='${GEMMA_MODEL_PATH}' SERVED_MODEL_NAME='gemma4-31b-final' PORT='18002' TP_SIZE='2' GPU_MEMORY_UTILIZATION='0.90' MAX_MODEL_LEN='32768' MAX_NUM_SEQS='16' NCCL_P2P_DISABLE='1' NCCL_IB_DISABLE='1' && bash scripts/start_gemma4_vllm.sh > '${LOG_DIR}/gemma_vllm.log' 2>&1"
fi

wait_for_server 18001 "Qwen3.6"
wait_for_server 18002 "Gemma4"

log "Step 5: run BreaKHis binary VLM suites for Qwen3.6 and Gemma4 in parallel"
bash scripts/run_breakhis_binary_vlm_suite.sh qwen > "${LOG_DIR}/breakhis_binary_qwen_suite.log" 2>&1 &
QWEN_BREAKHIS_PID=$!
bash scripts/run_breakhis_binary_vlm_suite.sh gemma > "${LOG_DIR}/breakhis_binary_gemma_suite.log" 2>&1 &
GEMMA_BREAKHIS_PID=$!

log "Step 6: start BreaKHis binary ResNet50 baseline"
(
  if [[ -n "${RESNET_CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${RESNET_CUDA_VISIBLE_DEVICES}"
  fi
  python scripts/train_resnet50_classification.py \
    --manifest-csv manifests/breakhis_binary_patient_split_seed3407.csv \
    --data-root data/raw/BreaKHis_v1_extracted \
    --output-dir outputs/final/resnet50_breakhis_binary_384_seed3407 \
    --image-size 384 \
    --batch-size 32 \
    --epochs 40 \
    --patience 8 \
    --num-workers 8 \
    --seed 3407 \
    --backbone-lr 3e-5 \
    --head-lr 1e-3 \
    --weight-decay 1e-4 \
    --selection-metric accuracy
) > "${LOG_DIR}/breakhis_binary_resnet50.log" 2>&1 &
RESNET_PID=$!

wait "${QWEN_BREAKHIS_PID}"
log "BreaKHis binary Qwen3.6 suite finished"
wait "${GEMMA_BREAKHIS_PID}"
log "BreaKHis binary Gemma4 suite finished"

log "Step 7: run DDR zero-shot for Qwen3.6 and Gemma4 in parallel"
bash scripts/run_ddr_zero_shot_suite.sh qwen > "${LOG_DIR}/ddr_qwen_zero_shot.log" 2>&1 &
QWEN_DDR_PID=$!
bash scripts/run_ddr_zero_shot_suite.sh gemma > "${LOG_DIR}/ddr_gemma_zero_shot.log" 2>&1 &
GEMMA_DDR_PID=$!

wait "${QWEN_DDR_PID}"
log "DDR Qwen3.6 zero-shot finished"
wait "${GEMMA_DDR_PID}"
log "DDR Gemma4 zero-shot finished"

wait "${RESNET_PID}"
log "BreaKHis binary ResNet50 finished"

if [[ "${KEEP_VLLM}" == "0" ]]; then
  tmux kill-session -t "${QWEN_SESSION}" 2>/dev/null || true
  tmux kill-session -t "${GEMMA_SESSION}" 2>/dev/null || true
  log "Stopped vLLM sessions"
else
  log "Keeping vLLM sessions running. Set KEEP_VLLM=0 to stop them at the end."
fi

log "Remote BreaKHis binary + DDR zero-shot run complete"
