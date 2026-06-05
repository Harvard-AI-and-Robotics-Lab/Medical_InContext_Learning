#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/final/qwen36_noicl_fixed6_random6.yaml
  configs/final/qwen36_clip_top6.yaml
  configs/final/qwen36_dinov3_cls_top6.yaml
  configs/final/qwen36_clip_dinov3cls_top6.yaml
  configs/final/gemma4_noicl_fixed6_random6.yaml
  configs/final/gemma4_clip_top6.yaml
  configs/final/gemma4_dinov3_cls_top6.yaml
  configs/final/gemma4_clip_dinov3cls_top6.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "Running ${cfg}"
  python scripts/run_final_classification.py --config "${cfg}"
done
