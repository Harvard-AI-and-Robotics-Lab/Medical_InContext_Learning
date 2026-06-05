#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
python3 scripts/train_resnet50_classification.py \
  --manifest-csv manifests/breakhis_patient_split_seed3407.csv \
  --data-root data/raw/BreaKHis_v1_extracted \
  --output-dir outputs/final/resnet50_breakhis_multiclass_384_seed3407 \
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
