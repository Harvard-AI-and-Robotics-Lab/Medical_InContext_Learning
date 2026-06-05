#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

FORCE_FEATURES="${FORCE_FEATURES:-0}"

CLIP_DIR="outputs/features_clip_global_ddr_1024/ddr/clip"
DINO_DIR="outputs/features_dinov3_global_ddr_1024/ddr/dinov3"
FUSED_DIR="outputs/features_clip_dinov3cls_05_global_ddr_1024/ddr/clip_dinov3cls05"
MANIFEST="${DDR_MANIFEST:-manifests/ddr_official_split_crop_pad_1024.csv}"

feature_complete() {
  local feature_dir="$1"
  python - "$feature_dir" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

feature_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
meta_path = feature_dir / "metadata.json"
emb_path = feature_dir / "global_embeddings.npy"
if not meta_path.exists() or not emb_path.exists():
    raise SystemExit(1)
meta = json.loads(meta_path.read_text())
emb = np.load(emb_path, mmap_mode="r")
expected = len(pd.read_csv(manifest))
if int(meta.get("n_samples", -1)) != expected or int(emb.shape[0]) != expected:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_or_skip_feature() {
  local name="$1"
  local cfg="$2"
  local out_dir="$3"
  if [[ "$FORCE_FEATURES" != "1" ]] && feature_complete "$out_dir"; then
    echo "[features] ${name} already complete: ${out_dir}"
    return
  fi
  echo "[features] extracting ${name}"
  python scripts/extract_features.py --config "$cfg"
}

run_or_skip_feature "CLIP global" "configs/final/ddr_extract_clip_global.yaml" "$CLIP_DIR"
run_or_skip_feature "DINOv3 CLS/global" "configs/final/ddr_extract_dinov3_global.yaml" "$DINO_DIR"

if [[ "$FORCE_FEATURES" != "1" ]] && feature_complete "$FUSED_DIR"; then
  echo "[features] fused CLIP+DINOv3 already complete: ${FUSED_DIR}"
else
  echo "[features] building fused CLIP+DINOv3 CLS features"
  python scripts/build_fused_features.py \
    --feature-a "$CLIP_DIR" \
    --feature-b "$DINO_DIR" \
    --output-dir "$FUSED_DIR" \
    --weight-a 0.5 \
    --weight-b 0.5 \
    --encoder-name clip_dinov3cls05 \
    --score-definition "0.5*CLIP_global_cosine + 0.5*DINOv3_CLS_cosine"
fi

echo "[features] DDR feature pipeline complete"
