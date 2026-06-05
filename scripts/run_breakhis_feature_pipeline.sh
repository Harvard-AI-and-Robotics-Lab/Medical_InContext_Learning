#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/extract_features.py --config configs/final/breakhis_extract_clip_global.yaml
python scripts/extract_features.py --config configs/final/breakhis_extract_dinov3_global.yaml

python scripts/build_fused_features.py \
  --feature-a outputs/features_clip_global_breakhis/breakhis/clip \
  --feature-b outputs/features_dinov3_global_breakhis/breakhis/dinov3 \
  --output-dir outputs/features_clip_dinov3cls_05_global_breakhis/breakhis/clip_dinov3cls05 \
  --encoder-name clip_dinov3cls05 \
  --score-definition "0.5*CLIP_global_cosine + 0.5*DINOv3_CLS_cosine"

python scripts/dump_breakhis_prompts.py \
  --features-dir outputs/features_clip_global_breakhis/breakhis/clip \
  --output-md outputs/smoke/breakhis_prompt_dump.md
