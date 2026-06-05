#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="${LOG_DIR:-logs/ddr_formal_1024}"
FEATURE_DIR="${FEATURE_DIR:-outputs/features_clip_dinov3cls_05_global_ddr_1024/ddr/clip_dinov3cls05}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

mkdir -p "$LOG_DIR"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "[wait] waiting for 1024 fused features at ${FEATURE_DIR}"
while [[ ! -f "${FEATURE_DIR}/metadata.json" || ! -f "${FEATURE_DIR}/global_embeddings.npy" ]]; do
  date
  sleep "$CHECK_INTERVAL"
done

echo "[vlm] starting Qwen3.6 and Gemma4 DDR 1024 suites"
bash scripts/run_ddr_vlm_suite.sh qwen > "${LOG_DIR}/qwen_vlm.log" 2>&1 &
QWEN_PID=$!
bash scripts/run_ddr_vlm_suite.sh gemma > "${LOG_DIR}/gemma_vlm.log" 2>&1 &
GEMMA_PID=$!

wait "$QWEN_PID"
echo "[vlm] Qwen3.6 DDR 1024 suite finished"
wait "$GEMMA_PID"
echo "[vlm] Gemma4 DDR 1024 suite finished"

echo "[vlm] DDR 1024 VLM suites complete"

if [[ "${RUN_RESNET_AFTER_VLM:-1}" == "1" ]]; then
  echo "[resnet] stopping DDR 1024 vLLM servers to free GPU memory"
  tmux kill-session -t ddr_qwen36_vllm_1024 2>/dev/null || true
  tmux kill-session -t ddr_gemma4_vllm_1024 2>/dev/null || true
  sleep 10

  echo "[resnet] starting DDR 1024 ResNet50 baseline"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${RESNET_CUDA_VISIBLE_DEVICES:-1}"
    python scripts/train_resnet50_classification.py \
      --manifest-csv manifests/ddr_official_split_crop_pad_1024.csv \
      --data-root . \
      --output-dir outputs/final/resnet50_ddr_crop_pad_1024_384_seed3407 \
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
  ) > "${LOG_DIR}/resnet50.log" 2>&1
  echo "[resnet] DDR 1024 ResNet50 baseline finished"
fi
