#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CLIP_GPU="${CLIP_GPU:-6}"
DINO_GPU="${DINO_GPU:-7}"

CUDA_VISIBLE_DEVICES="${CLIP_GPU}" \
  python3 scripts/extract_features.py \
    --config configs/final/vqamed2019_extract_clip_global.yaml \
    --datasets vqamed2019 \
    --encoders clip

CUDA_VISIBLE_DEVICES="${DINO_GPU}" \
  python3 scripts/extract_features.py \
    --config configs/final/vqamed2019_extract_dinov3_global.yaml \
    --datasets vqamed2019 \
    --encoders dinov3

python3 scripts/build_fused_features.py \
  --feature-a outputs/features_clip_global_vqamed2019/vqamed2019/clip \
  --feature-b outputs/features_dinov3_global_vqamed2019/vqamed2019/dinov3 \
  --output-dir outputs/features_clip_dinov3cls_05_global_vqamed2019/vqamed2019/clip_dinov3cls05 \
  --weight-a 0.5 \
  --weight-b 0.5 \
  --encoder-name clip_dinov3cls05
