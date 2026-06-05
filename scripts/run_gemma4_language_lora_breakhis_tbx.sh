#!/usr/bin/env bash
set -euo pipefail

cd /data/home/mindazhao/Incontext_Learning_Git
mkdir -p logs/runs outputs/lora

PY=/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin/python
MODEL=/data/home/mindazhao/hf_models/google_gemma-4-31B-it

tmux new-session -d -s gemma4_lora_breakhis_binary \
  "bash -lc 'cd /data/home/mindazhao/Incontext_Learning_Git && CUDA_VISIBLE_DEVICES=3,4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 ${PY} scripts/train_gemma4_language_lora.py --dataset breakhis_binary --model_path ${MODEL} --output_dir outputs/lora/gemma4_language_lora_breakhis_binary_r16_a32_lr1e-4_seed3407 --epochs 3 --batch_size 1 --grad_accum 16 --lr 1e-4 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --eval_steps 200 --eval_subset 200 --logging_steps 10' > logs/runs/gemma4_lora_breakhis_binary.log 2>&1"

tmux new-session -d -s gemma4_lora_tbx11k \
  "bash -lc 'cd /data/home/mindazhao/Incontext_Learning_Git && CUDA_VISIBLE_DEVICES=5,6 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 ${PY} scripts/train_gemma4_language_lora.py --dataset tbx11k --model_path ${MODEL} --output_dir outputs/lora/gemma4_language_lora_tbx11k_r16_a32_lr1e-4_seed3407 --epochs 3 --batch_size 1 --grad_accum 16 --lr 1e-4 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --eval_steps 200 --eval_subset 200 --logging_steps 10' > logs/runs/gemma4_lora_tbx11k.log 2>&1"

tmux ls | grep 'gemma4_lora_'
