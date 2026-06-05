#!/usr/bin/env bash
set -euo pipefail

cd /data/home/mindazhao/Incontext_Learning_Git

PY=/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin/python
export PATH=/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin:${PATH}

CUDA_DEVICES=${CUDA_DEVICES:-6,7}
PORT=${PORT:-18010}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-lag-lora-merged}
API_KEY=${API_KEY:-EMPTY}

BASE_MODEL=/data/home/mindazhao/hf_models/google_gemma-4-31B-it
ADAPTER_DIR=outputs/lora/gemma4_language_lora_lag_project_r16_a32_lr1e-4_seed3407/best
MERGED_DIR=outputs/merged/gemma4_lag_lora_merged
OUT_DIR=outputs/lora/gemma4_language_lora_lag_project_r16_a32_lr1e-4_seed3407

LOG_DIR=logs/vllm
RUN_LOG=${LOG_DIR}/lag_lora_test512_merged_run.log
MERGE_LOG=${LOG_DIR}/lag_lora_test512_merge.log
SERVER_LOG=${LOG_DIR}/lag_lora_test512_server.log
EVAL_LOG=${LOG_DIR}/lag_lora_test512_eval.log

mkdir -p "${LOG_DIR}" "${MERGED_DIR}" "${OUT_DIR}"

echo "[start] $(date)" | tee "${RUN_LOG}"
echo "[config] CUDA_DEVICES=${CUDA_DEVICES} PORT=${PORT} MODEL=${SERVED_MODEL_NAME}" | tee -a "${RUN_LOG}"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[merge] creating ${MERGED_DIR}" | tee -a "${RUN_LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PY}" scripts/merge_gemma4_language_lora.py \
    --base_model "${BASE_MODEL}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --output_dir "${MERGED_DIR}" \
    --max_memory_per_gpu 46GiB \
    --cpu_memory 700GiB \
    --max_shard_size 10GB \
    > "${MERGE_LOG}" 2>&1
else
  echo "[merge] existing merged model found, skipping" | tee -a "${RUN_LOG}"
fi

echo "[serve] launching vLLM on ${CUDA_DEVICES}:${PORT}" | tee -a "${RUN_LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
VLLM_API_KEY="${API_KEY}" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
vllm serve "${MERGED_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":1}' \
  --generation-config vllm \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping vLLM pid=${SERVER_PID}" | tee -a "${RUN_LOG}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[serve] waiting for API" | tee -a "${RUN_LOG}"
for _ in $(seq 1 180); do
  if curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serve] ready" | tee -a "${RUN_LOG}"
    break
  fi
  sleep 10
done
curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null

echo "[eval] LAG test generation max_tokens=512" | tee -a "${RUN_LOG}"
CUDA_VISIBLE_DEVICES="" VLLM_API_KEY="${API_KEY}" "${PY}" scripts/eval_gemma4_lora_vllm_api.py \
  --dataset lag_project \
  --model "${SERVED_MODEL_NAME}" \
  --base_url "http://127.0.0.1:${PORT}/v1" \
  --output_json "${OUT_DIR}/test_generation_vllm_merged_max512_metrics.json" \
  --predictions_json "${OUT_DIR}/test_generation_vllm_merged_max512_predictions.json" \
  --split test \
  --seed 3407 \
  --batch_size 32 \
  --max_workers 32 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --image_max_side -1 \
  > "${EVAL_LOG}" 2>&1

echo "[done] $(date)" | tee -a "${RUN_LOG}"
