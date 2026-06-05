#!/usr/bin/env bash
set -euo pipefail

cd /data/home/mindazhao/Incontext_Learning_Git

PY=${PY:-/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin/python}
export PATH=/data/home/mindazhao/miniforge3/envs/qwen36_vllm/bin:${PATH}

CUDA_DEVICES=${CUDA_DEVICES:-0,1}
PORT=${PORT:-18216}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-gemma4-slake-vqa-lora-assistant512}
API_KEY=${API_KEY:-EMPTY}

BASE_MODEL=${BASE_MODEL:-/data/home/mindazhao/hf_models/google_gemma-4-31B-it}
OUT_DIR=${OUT_DIR:-outputs/lora/gemma4_slake_vqa_lora_assistant_completion_r16_a32_lr1e-4_seed3407_max512}
ADAPTER_DIR="${OUT_DIR}/best"
MERGED_DIR="${OUT_DIR}/best_merged"
TEST_OUT_DIR="${OUT_DIR}/test_generation_vllm_merged_max512"

LOG_DIR=logs/slake_lora_assistant512
RUN_LOG="${LOG_DIR}/pipeline.log"
TRAIN_LOG="${LOG_DIR}/train.log"
MERGE_LOG="${LOG_DIR}/merge.log"
SERVER_LOG="${LOG_DIR}/vllm_server.log"
EVAL_LOG="${LOG_DIR}/test_generation_vllm_merged_max512.log"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping vLLM pid=${SERVER_PID}" | tee -a "${RUN_LOG}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[start] $(date)" | tee "${RUN_LOG}"
echo "[config] CUDA_DEVICES=${CUDA_DEVICES} PORT=${PORT} OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"

echo "[train] assistant-completion LoRA, lr=1e-4, eval/test max_tokens=512" | tee -a "${RUN_LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
"${PY}" scripts/train_gemma4_vqa_lora.py \
  --dataset slake \
  --model_path "${BASE_MODEL}" \
  --output_dir "${OUT_DIR}" \
  --epochs 5 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --eval_steps 100 \
  --eval_subset 200 \
  --eval_stratified \
  --eval_max_new_tokens 512 \
  --loss_mode assistant_completion \
  --selection_metric balanced_open_closed \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0 \
  --logging_steps 10 \
  > "${TRAIN_LOG}" 2>&1

echo "[merge] creating merged model at ${MERGED_DIR}" | tee -a "${RUN_LOG}"
rm -rf "${MERGED_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
"${PY}" scripts/merge_gemma4_language_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --output_dir "${MERGED_DIR}" \
  --max_memory_per_gpu 46GiB \
  --cpu_memory 700GiB \
  --max_shard_size 10GB \
  > "${MERGE_LOG}" 2>&1

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

echo "[serve] waiting for API" | tee -a "${RUN_LOG}"
for _ in $(seq 1 180); do
  if curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serve] ready" | tee -a "${RUN_LOG}"
    break
  fi
  sleep 10
done
curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null

echo "[eval] SLAKE test generation via merged vLLM, max_tokens=512" | tee -a "${RUN_LOG}"
rm -rf "${TEST_OUT_DIR}"
CUDA_VISIBLE_DEVICES="" \
VLLM_API_KEY="${API_KEY}" \
"${PY}" scripts/eval_gemma4_vqa_lora_vllm_api.py \
  --dataset slake \
  --model "${SERVED_MODEL_NAME}" \
  --base_url "http://127.0.0.1:${PORT}/v1" \
  --output_dir "${TEST_OUT_DIR}" \
  --split test \
  --seed 3407 \
  --batch_size 128 \
  --max_workers 128 \
  --max_tokens 512 \
  --temperature 0.0 \
  --top_p 1.0 \
  --image_max_side -1 \
  --response_format json_object \
  > "${EVAL_LOG}" 2>&1

echo "[done] $(date)" | tee -a "${RUN_LOG}"
