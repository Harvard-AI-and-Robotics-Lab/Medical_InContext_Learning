#!/usr/bin/env bash
set -euo pipefail
cd /data/home/mindazhao/Incontext_Learning_Git
export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
export PATH=/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin:$PATH
echo "[qwen_resume] started $(date)"
python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_noicl_fixed6_random6.yaml --methods random_icl fixed_random_6
python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_clip_top6.yaml
python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_dinov3_cls_top6.yaml
python3 scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_clip_dinov3cls_top6.yaml
echo "[qwen_resume] finished $(date)"
