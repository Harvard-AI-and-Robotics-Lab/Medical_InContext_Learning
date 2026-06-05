#!/usr/bin/env bash
set -euo pipefail

LOCAL_MODEL_PATH="/data/home/mindazhao/hf_models/google_gemma-4-31B-it"
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${LOCAL_MODEL_PATH}" ]]; then
    MODEL_PATH="${LOCAL_MODEL_PATH}"
  else
    MODEL_PATH="google/gemma-4-31B-it"
  fi
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4-31b-final}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18002}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

VLLM_ARGS=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":8}'
)

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  VLLM_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  VLLM_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${GENERATION_CONFIG}" ]]; then
  VLLM_ARGS+=(--generation-config "${GENERATION_CONFIG}")
fi

"${VLLM_ARGS[@]}"
