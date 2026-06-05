#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/prepare_vqa_rad_manifest.py \
  --metadata-csv data/VQA_RAD/metadata.csv \
  --data-root data/VQA_RAD \
  --output manifests/vqa_rad_official_split.csv

python3 scripts/preprocess_vqa_rad_320.py \
  --manifest-csv manifests/vqa_rad_official_split.csv \
  --data-root . \
  --output-root data/processed/vqa_rad_512 \
  --output-manifest manifests/vqa_rad_official_split_512.csv \
  --size 512 \
  --quality 95 \
  --workers "${VQA_RAD_PREPROCESS_WORKERS:-16}" \
  --skip-existing

if [ "${SKIP_FEATURES:-0}" = "1" ]; then
  echo "[vqa_rad] SKIP_FEATURES=1, stopping after manifest/image preprocessing."
  exit 0
fi

python3 scripts/extract_features.py --config configs/final/vqa_rad_extract_clip_global_512.yaml
python3 scripts/extract_features.py --config configs/final/vqa_rad_extract_dinov3_global_512.yaml

python3 scripts/build_fused_features.py \
  --feature-a outputs/features_clip_global_vqa_rad_512/vqa_rad/clip \
  --feature-b outputs/features_dinov3_global_vqa_rad_512/vqa_rad/dinov3 \
  --output-dir outputs/features_clip_dinov3cls_05_global_vqa_rad_512/vqa_rad/clip_dinov3cls05 \
  --weight-a 0.5 \
  --weight-b 0.5 \
  --encoder-name clip_dinov3cls05

echo "[vqa_rad] 512 preprocessing and feature pipeline complete"
