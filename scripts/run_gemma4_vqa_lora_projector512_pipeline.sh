#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET=${DATASET:?Set DATASET to pathvqa, vqamed2019, or vqa_rad}
PY=${PY:-python3}
if [[ -n "${VLLM_BIN_DIR:-}" ]]; then
  export PATH="${VLLM_BIN_DIR}:${PATH}"
fi

CUDA_DEVICES=${CUDA_DEVICES:-0,1}
PORT=${PORT:-18326}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-gemma4-${DATASET}-vqa-lora-projector512}
API_KEY=${API_KEY:-EMPTY}

BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
OUT_DIR=${OUT_DIR:-outputs/lora/gemma4_${DATASET}_vqa_lora_language_projector_r16_a32_lr1e-4_seed3407_max512}
ADAPTER_DIR="${OUT_DIR}/best"
MERGED_DIR="${OUT_DIR}/best_merged"
TEST_OUT_DIR="${OUT_DIR}/test_generation_vllm_merged_max512"

LOG_DIR=${LOG_DIR:-logs/${DATASET}_lora_projector512}
RUN_LOG="${LOG_DIR}/pipeline.log"
TRAIN_LOG="${LOG_DIR}/train.log"
MERGE_LOG="${LOG_DIR}/merge.log"
SERVER_LOG="${LOG_DIR}/vllm_server.log"
EVAL_LOG="${LOG_DIR}/test_generation_vllm_merged_max512.log"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

TRAIN_PROMPT_ARGS=()
EVAL_PROMPT_ARGS=()
if [[ ${VQA_SYSTEM_PROMPT+x} ]]; then
  TRAIN_PROMPT_ARGS+=(--vqa_system_prompt "${VQA_SYSTEM_PROMPT}")
  EVAL_PROMPT_ARGS+=(--vqa_system_prompt "${VQA_SYSTEM_PROMPT}")
fi
if [[ ${VQA_QUERY_TEMPLATE+x} ]]; then
  TRAIN_PROMPT_ARGS+=(--vqa_query_template "${VQA_QUERY_TEMPLATE}")
  EVAL_PROMPT_ARGS+=(--vqa_query_template "${VQA_QUERY_TEMPLATE}")
fi
if [[ ${JSON_INSTRUCTION+x} ]]; then
  TRAIN_PROMPT_ARGS+=(--json_instruction "${JSON_INSTRUCTION}")
  EVAL_PROMPT_ARGS+=(--json_instruction "${JSON_INSTRUCTION}")
fi
if [[ ${ASSISTANT_FORMAT+x} ]]; then
  TRAIN_PROMPT_ARGS+=(--assistant_format "${ASSISTANT_FORMAT}")
fi
EVAL_RESPONSE_FORMAT=${EVAL_RESPONSE_FORMAT:-json_object}

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
echo "[config] DATASET=${DATASET} CUDA_DEVICES=${CUDA_DEVICES} PORT=${PORT} OUT_DIR=${OUT_DIR}" | tee -a "${RUN_LOG}"
echo "[config] ASSISTANT_FORMAT=${ASSISTANT_FORMAT:-json} EVAL_RESPONSE_FORMAT=${EVAL_RESPONSE_FORMAT}" | tee -a "${RUN_LOG}"
echo "[train] ${DATASET} language+projector LoRA, lr=1e-4, eval/test max_tokens=512" | tee -a "${RUN_LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
"${PY}" scripts/train_gemma4_vqa_lora.py \
  --dataset "${DATASET}" \
  --model_path "${BASE_MODEL}" \
  --output_dir "${OUT_DIR}" \
  --epochs 5 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_scope language_projector \
  --eval_steps 100 \
  --eval_subset 200 \
  --eval_stratified \
  --eval_max_new_tokens 512 \
  --loss_mode assistant_completion \
  --selection_metric balanced_open_token_recall_closed \
  --early_stop_patience 3 \
  --early_stop_min_delta 0.0 \
  --logging_steps 10 \
  "${TRAIN_PROMPT_ARGS[@]}" \
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

echo "[eval] ${DATASET} test generation via merged vLLM, max_tokens=512" | tee -a "${RUN_LOG}"
rm -rf "${TEST_OUT_DIR}"
CUDA_VISIBLE_DEVICES="" \
VLLM_API_KEY="${API_KEY}" \
"${PY}" scripts/eval_gemma4_vqa_lora_vllm_api.py \
  --dataset "${DATASET}" \
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
  --response_format "${EVAL_RESPONSE_FORMAT}" \
  "${EVAL_PROMPT_ARGS[@]}" \
  > "${EVAL_LOG}" 2>&1

echo "[done] $(date)" | tee -a "${RUN_LOG}"
