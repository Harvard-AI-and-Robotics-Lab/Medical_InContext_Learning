#!/usr/bin/env bash
set -euo pipefail
cd /data/home/mindazhao/Incontext_Learning_Git
mkdir -p logs/runs
export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
export PATH=/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin:$PATH

echo "[driver] started $(date)"

(
  echo "[qwen noicl] started $(date)"
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_noicl_fixed6_random6.yaml
  echo "[qwen noicl] finished $(date)"
) > logs/runs/tbx11k_qwen36_noicl_fixed6_random6.log 2>&1 &
qwen_noicl_pid=$!

(
  echo "[gemma noicl] started $(date)"
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_noicl_fixed6_random6.yaml
  echo "[gemma noicl] finished $(date)"
) > logs/runs/tbx11k_gemma4_noicl_fixed6_random6.log 2>&1 &
gemma_noicl_pid=$!

(
  echo "[clip features] started $(date)"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_features.py --config configs/final/tbx11k_extract_clip_global.yaml
  echo "[clip features] finished $(date)"
) > logs/runs/tbx11k_clip_features.log 2>&1 &
clip_pid=$!

(
  echo "[dinov3 features] started $(date)"
  CUDA_VISIBLE_DEVICES=7 python3 scripts/extract_features.py --config configs/final/tbx11k_extract_dinov3_global.yaml
  echo "[dinov3 features] finished $(date)"
) > logs/runs/tbx11k_dinov3_features.log 2>&1 &
dino_pid=$!

wait $clip_pid
wait $dino_pid

echo "[fused features] started $(date)" > logs/runs/tbx11k_fused_features.log
python3 scripts/build_fused_features.py   --feature-a outputs/features_clip_global_tbx11k/tbx11k/clip   --feature-b outputs/features_dinov3_global_tbx11k/tbx11k/dinov3   --output-dir outputs/features_clip_dinov3cls_05_global_tbx11k/tbx11k/clip_dinov3cls05   --weight-a 0.5   --weight-b 0.5   --encoder-name clip_dinov3cls05 >> logs/runs/tbx11k_fused_features.log 2>&1
echo "[fused features] finished $(date)" >> logs/runs/tbx11k_fused_features.log

(
  wait $qwen_noicl_pid
  echo "[qwen retrieval] started $(date)"
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_clip_top6.yaml
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_dinov3_cls_top6.yaml
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_clip_dinov3cls_top6.yaml
  echo "[qwen retrieval] finished $(date)"
) > logs/runs/tbx11k_qwen36_retrieval.log 2>&1 &
qwen_ret_pid=$!

(
  wait $gemma_noicl_pid
  echo "[gemma retrieval] started $(date)"
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_clip_top6.yaml
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_dinov3_cls_top6.yaml
  python3 scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_clip_dinov3cls_top6.yaml
  echo "[gemma retrieval] finished $(date)"
) > logs/runs/tbx11k_gemma4_retrieval.log 2>&1 &
gemma_ret_pid=$!

wait $qwen_ret_pid
wait $gemma_ret_pid

echo "[driver] finished $(date)"
