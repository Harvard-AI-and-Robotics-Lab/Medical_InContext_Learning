#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs

python3 scripts/preprocess_ddr_images.py \
  --manifest-csv manifests/ddr_official_split.csv \
  --data-root . \
  --output-root data/processed/ddr_crop_pad_512 \
  --output-manifest manifests/ddr_official_split_crop_pad_512.csv \
  --summary-json manifests/ddr_official_split_crop_pad_512.summary.json \
  --size 512 \
  --threshold 10 \
  --margin 8 \
  --quality 95 \
  --workers 16 \
  --skip-existing

wait_for_free_gpu() {
  while true; do
    for gpu in ${FEATURE_GPU_CANDIDATES:-7 2 1}; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      if [ "$used" -lt 2000 ]; then
        echo "$gpu"
        return 0
      fi
    done
    echo "[$(date)] waiting for a free feature GPU among: ${FEATURE_GPU_CANDIDATES:-7 2 1}"
    sleep 60
  done
}

need_features() {
  local meta="$1"
  python3 - "$meta" <<'END_NEEDED'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(0)
try:
    m = json.load(open(p))
except Exception:
    raise SystemExit(0)
raise SystemExit(1 if int(m.get("n_samples", 0)) == 13673 else 0)
END_NEEDED
}

if need_features outputs/features_clip_global_ddr_512/ddr/clip/metadata.json; then
  gpu=$(wait_for_free_gpu)
  echo "[$(date)] extracting DDR 512 CLIP on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/extract_features.py --config configs/final/ddr_extract_clip_global_512.yaml
else
  echo "[$(date)] DDR 512 CLIP features already complete"
fi

if need_features outputs/features_dinov3_global_ddr_512/ddr/dinov3/metadata.json; then
  gpu=$(wait_for_free_gpu)
  echo "[$(date)] extracting DDR 512 DINOv3 on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/extract_features.py --config configs/final/ddr_extract_dinov3_global_512.yaml
else
  echo "[$(date)] DDR 512 DINOv3 features already complete"
fi

if need_features outputs/features_clip_dinov3cls_05_global_ddr_512/ddr/clip_dinov3cls05/metadata.json; then
  echo "[$(date)] building DDR 512 CLIP+DINOv3 0.5/0.5 fused features"
  python3 scripts/build_fused_features.py \
    --feature-a outputs/features_clip_global_ddr_512/ddr/clip \
    --feature-b outputs/features_dinov3_global_ddr_512/ddr/dinov3 \
    --output-dir outputs/features_clip_dinov3cls_05_global_ddr_512/ddr/clip_dinov3cls05 \
    --weight-a 0.5 \
    --weight-b 0.5 \
    --encoder-name clip_dinov3cls05
else
  echo "[$(date)] DDR 512 fused features already complete"
fi

if ! tmux has-session -t ddr_512_qwen36_vlm 2>/dev/null; then
  tmux new-session -d -s ddr_512_qwen36_vlm "bash -lc 'cd \"${REPO_ROOT}\"; export VLLM_API_KEY=EMPTY; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_noicl_fixed6_random6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_dinov3_cls_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_dinov3cls_top6_512.yaml' > logs/runs/ddr_512_qwen36_vlm.log 2>&1"
fi

if ! tmux has-session -t ddr_512_gemma4_vlm 2>/dev/null; then
  tmux new-session -d -s ddr_512_gemma4_vlm "bash -lc 'cd \"${REPO_ROOT}\"; export VLLM_API_KEY=EMPTY; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_noicl_fixed6_random6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_dinov3_cls_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_dinov3cls_top6_512.yaml' > logs/runs/ddr_512_gemma4_vlm.log 2>&1"
fi

echo "[$(date)] DDR 512 prepare/features done; VLM sessions launched or already present."
