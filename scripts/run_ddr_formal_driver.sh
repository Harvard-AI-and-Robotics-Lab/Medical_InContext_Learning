#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/ddr_formal_${RUN_ID}}"
FEATURE_GPU_UUID="${FEATURE_GPU_UUID:-GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb}"
RESNET_GPU_UUID="${RESNET_GPU_UUID:-$FEATURE_GPU_UUID}"
RAW_DDR_MANIFEST="${RAW_DDR_MANIFEST:-manifests/ddr_official_split.csv}"
DDR_MANIFEST="${DDR_MANIFEST:-manifests/ddr_official_split_crop_pad_1024.csv}"
DDR_PREPROCESSED_ROOT="${DDR_PREPROCESSED_ROOT:-data/processed/ddr_crop_pad_1024}"

mkdir -p "$LOG_DIR"
echo "[driver] run_id=${RUN_ID}"
echo "[driver] log_dir=${LOG_DIR}"
echo "[driver] feature_gpu_uuid=${FEATURE_GPU_UUID}"
echo "[driver] resnet_gpu_uuid=${RESNET_GPU_UUID}"
echo "[driver] raw_ddr_manifest=${RAW_DDR_MANIFEST}"
echo "[driver] ddr_manifest=${DDR_MANIFEST}"
echo "[driver] ddr_preprocessed_root=${DDR_PREPROCESSED_ROOT}"

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export HF_HOME="${HF_HOME:-/shared/ssd_14T/home/mindazhao/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

echo "[driver] checking existing vLLM endpoints"
curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" http://127.0.0.1:18001/v1/models > "$LOG_DIR/qwen_models.json"
curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" http://127.0.0.1:18002/v1/models > "$LOG_DIR/gemma_models.json"

echo "[driver] preprocessing DDR images to crop/pad 1024"
python scripts/preprocess_ddr_images.py \
  --manifest-csv "$RAW_DDR_MANIFEST" \
  --data-root . \
  --output-root "$DDR_PREPROCESSED_ROOT" \
  --output-manifest "$DDR_MANIFEST" \
  --summary-json "${DDR_MANIFEST%.csv}.summary.json" \
  --size 1024 \
  --threshold 10 \
  --margin 8 \
  --quality 95 \
  --workers "${DDR_PREPROCESS_WORKERS:-16}" \
  --skip-existing > "$LOG_DIR/preprocess_ddr_1024.log" 2>&1

echo "[driver] starting feature pipeline"
(
  export CUDA_VISIBLE_DEVICES="$FEATURE_GPU_UUID"
  export DDR_MANIFEST="$DDR_MANIFEST"
  bash scripts/run_ddr_feature_pipeline.sh
) 2>&1 | tee "$LOG_DIR/features.log"

echo "[driver] starting Qwen3.6 and Gemma4 VLM suites"
bash scripts/run_ddr_vlm_suite.sh qwen > "$LOG_DIR/qwen_vlm.log" 2>&1 &
QWEN_PID=$!
bash scripts/run_ddr_vlm_suite.sh gemma > "$LOG_DIR/gemma_vlm.log" 2>&1 &
GEMMA_PID=$!

echo "[driver] starting ResNet50 fine-tune"
(
  export CUDA_VISIBLE_DEVICES="$RESNET_GPU_UUID"
  python scripts/train_resnet50_classification.py \
    --manifest-csv "$DDR_MANIFEST" \
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
) > "$LOG_DIR/resnet50.log" 2>&1 &
RESNET_PID=$!

wait "$QWEN_PID"
echo "[driver] Qwen3.6 VLM suite finished"
wait "$GEMMA_PID"
echo "[driver] Gemma4 VLM suite finished"
wait "$RESNET_PID"
echo "[driver] ResNet50 finished"

echo "[driver] DDR formal run complete"
