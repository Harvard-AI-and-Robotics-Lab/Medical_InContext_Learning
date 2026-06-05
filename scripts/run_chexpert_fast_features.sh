#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs

expected_rows=224316
features_ready() {
  python3 - "$1" "$expected_rows" <<'PYCHECK'
import json, sys
p=sys.argv[1]; expected=int(sys.argv[2])
try:
    meta=json.load(open(p))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if int(meta.get('n_samples', -1))==expected else 1)
PYCHECK
}

run_sharded_features() {
  pids=()
  if ! features_ready outputs/features_clip_global_chexpert_320_fast/chexpert/clip/metadata.json; then
    echo "[$(date)] extracting CheXpert CLIP fast shards on GPU0/GPU1"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_clip_features_shard_fast.py \
      --config configs/final/chexpert_extract_clip_global_320_fast.yaml \
      --shard-index 0 --num-shards 2 \
      --output-root outputs/features_clip_global_chexpert_320_fast_shards \
      --batch-size 512 --num-workers 16 \
      > logs/runs/chexpert_fast_clip_shard0.log 2>&1 &
    pids+=("$!")
    CUDA_VISIBLE_DEVICES=1 python3 scripts/extract_clip_features_shard_fast.py \
      --config configs/final/chexpert_extract_clip_global_320_fast.yaml \
      --shard-index 1 --num-shards 2 \
      --output-root outputs/features_clip_global_chexpert_320_fast_shards \
      --batch-size 512 --num-workers 16 \
      > logs/runs/chexpert_fast_clip_shard1.log 2>&1 &
    pids+=("$!")
  else
    echo "[$(date)] CheXpert CLIP fast features already complete"
  fi

  if ! features_ready outputs/features_dinov3_global_chexpert_320_fast/chexpert/dinov3/metadata.json; then
    echo "[$(date)] extracting CheXpert DINOv3 fast shards on GPU0/GPU1/GPU2"
    CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_features_shard.py \
      --config configs/final/chexpert_extract_dinov3_global_320_fast.yaml \
      --shard-index 0 --num-shards 3 \
      --output-root outputs/features_dinov3_global_chexpert_320_fast_shards \
      > logs/runs/chexpert_fast_dinov3_shard0.log 2>&1 &
    pids+=("$!")
    CUDA_VISIBLE_DEVICES=1 python3 scripts/extract_features_shard.py \
      --config configs/final/chexpert_extract_dinov3_global_320_fast.yaml \
      --shard-index 1 --num-shards 3 \
      --output-root outputs/features_dinov3_global_chexpert_320_fast_shards \
      > logs/runs/chexpert_fast_dinov3_shard1.log 2>&1 &
    pids+=("$!")
    CUDA_VISIBLE_DEVICES=2 python3 scripts/extract_features_shard.py \
      --config configs/final/chexpert_extract_dinov3_global_320_fast.yaml \
      --shard-index 2 --num-shards 3 \
      --output-root outputs/features_dinov3_global_chexpert_320_fast_shards \
      > logs/runs/chexpert_fast_dinov3_shard2.log 2>&1 &
    pids+=("$!")
  else
    echo "[$(date)] CheXpert DINOv3 fast features already complete"
  fi

  if [ "${#pids[@]}" -gt 0 ]; then
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  fi

  if ! features_ready outputs/features_clip_global_chexpert_320_fast/chexpert/clip/metadata.json; then
    echo "[$(date)] merging CLIP shards"
    python3 scripts/merge_feature_shards.py \
      --shard-dirs \
      outputs/features_clip_global_chexpert_320_fast_shards/shard_00/chexpert/clip \
      outputs/features_clip_global_chexpert_320_fast_shards/shard_01/chexpert/clip \
      --output-dir outputs/features_clip_global_chexpert_320_fast/chexpert/clip
  fi

  if ! features_ready outputs/features_dinov3_global_chexpert_320_fast/chexpert/dinov3/metadata.json; then
    echo "[$(date)] merging DINOv3 shards"
    python3 scripts/merge_feature_shards.py \
      --shard-dirs \
      outputs/features_dinov3_global_chexpert_320_fast_shards/shard_00/chexpert/dinov3 \
      outputs/features_dinov3_global_chexpert_320_fast_shards/shard_01/chexpert/dinov3 \
      outputs/features_dinov3_global_chexpert_320_fast_shards/shard_02/chexpert/dinov3 \
      --output-dir outputs/features_dinov3_global_chexpert_320_fast/chexpert/dinov3
  fi
}

run_sharded_features

if ! features_ready outputs/features_clip_dinov3cls_05_global_chexpert_320_fast/chexpert/clip_dinov3cls05/metadata.json; then
  echo "[$(date)] building CheXpert CLIP+DINOv3 0.5/0.5 fused fast features"
  python3 scripts/build_fused_features.py \
    --feature-a outputs/features_clip_global_chexpert_320_fast/chexpert/clip \
    --feature-b outputs/features_dinov3_global_chexpert_320_fast/chexpert/dinov3 \
    --output-dir outputs/features_clip_dinov3cls_05_global_chexpert_320_fast/chexpert/clip_dinov3cls05 \
    --weight-a 0.5 \
    --weight-b 0.5 \
    --encoder-name clip_dinov3cls05
fi

start_vlm_queue() {
  local session="$1"
  local wait_session="$2"
  shift 2
  local configs=("$@")
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[$(date)] ${session} already running"
    return 0
  fi
  local cmd="cd \"${REPO_ROOT}\"; export VLLM_API_KEY=EMPTY; while tmux has-session -t ${wait_session} 2>/dev/null; do echo '[wait] waiting for ${wait_session} to finish'; sleep 300; done;"
  for cfg in "${configs[@]}"; do
    cmd+=" python3 scripts/run_final_classification.py --config ${cfg};"
  done
  tmux new-session -d -s "$session" "bash -lc \"${cmd}\" > logs/runs/${session}.log 2>&1"
  echo "[$(date)] started ${session}"
}

start_vlm_queue chexpert_qwen36_vlm ddr_512_qwen36_vlm \
  configs/final/chexpert5_qwen36_noicl_fixed6_random6_320.yaml \
  configs/final/chexpert5_qwen36_clip_top6_320.yaml \
  configs/final/chexpert5_qwen36_dinov3_cls_top6_320.yaml \
  configs/final/chexpert5_qwen36_clip_dinov3cls_top6_320.yaml

start_vlm_queue chexpert_gemma4_vlm ddr_512_gemma4_vlm \
  configs/final/chexpert5_gemma4_noicl_fixed6_random6_320.yaml \
  configs/final/chexpert5_gemma4_clip_top6_320.yaml \
  configs/final/chexpert5_gemma4_dinov3_cls_top6_320.yaml \
  configs/final/chexpert5_gemma4_clip_dinov3cls_top6_320.yaml

echo "[$(date)] CheXpert fast feature pipeline setup complete"
