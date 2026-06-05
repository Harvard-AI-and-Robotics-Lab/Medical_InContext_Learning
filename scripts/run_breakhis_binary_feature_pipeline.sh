#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

FORCE_FEATURES="${FORCE_FEATURES:-0}"
MAGS=(40 100 200 400)

feature_complete() {
  local feature_dir="$1"
  local manifest="$2"
  python - "$feature_dir" "$manifest" <<'PY'
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
  local manifest="$4"
  if [[ "$FORCE_FEATURES" != "1" ]] && feature_complete "$out_dir" "$manifest"; then
    echo "[features] ${name} already complete: ${out_dir}"
    return
  fi
  echo "[features] extracting ${name}"
  python scripts/extract_features.py --config "$cfg"
}

for mag in "${MAGS[@]}"; do
  manifest="manifests/breakhis_binary_patient_split_seed3407_mag${mag}.csv"
  clip_cfg="configs/final/breakhis_binary_mag${mag}_extract_clip_global.yaml"
  dino_cfg="configs/final/breakhis_binary_mag${mag}_extract_dinov3_global.yaml"
  clip_dir="outputs/features_clip_global_breakhis_binary_mag${mag}/breakhis_binary/clip"
  dino_dir="outputs/features_dinov3_global_breakhis_binary_mag${mag}/breakhis_binary/dinov3"
  fused_dir="outputs/features_clip_dinov3cls_05_global_breakhis_binary_mag${mag}/breakhis_binary/clip_dinov3cls05"

  run_or_skip_feature "BreaKHis binary mag${mag} CLIP global" "$clip_cfg" "$clip_dir" "$manifest"
  run_or_skip_feature "BreaKHis binary mag${mag} DINOv3 CLS/global" "$dino_cfg" "$dino_dir" "$manifest"

  if [[ "$FORCE_FEATURES" != "1" ]] && feature_complete "$fused_dir" "$manifest"; then
    echo "[features] BreaKHis binary mag${mag} fused CLIP+DINOv3 already complete: ${fused_dir}"
  else
    echo "[features] building BreaKHis binary mag${mag} fused CLIP+DINOv3 CLS features"
    python scripts/build_fused_features.py \
      --feature-a "$clip_dir" \
      --feature-b "$dino_dir" \
      --output-dir "$fused_dir" \
      --weight-a 0.5 \
      --weight-b 0.5 \
      --encoder-name clip_dinov3cls05 \
      --score-definition "0.5*CLIP_global_cosine + 0.5*DINOv3_CLS_cosine"
  fi
done

echo "[features] BreaKHis binary magnification-specific feature pipeline complete"
