#!/usr/bin/env bash
set -euo pipefail

LOCAL_MODEL_PATH="/data/home/mindazhao/hf_models/Qwen_Qwen3.6-27B"
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${LOCAL_MODEL_PATH}" ]]; then
    MODEL_PATH="${LOCAL_MODEL_PATH}"
  else
    MODEL_PATH="Qwen/Qwen3.6-27B"
  fi
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b-final}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18003}"
TP_SIZE="${TP_SIZE:-2}"
MIN_VLLM_VERSION="${MIN_VLLM_VERSION:-0.17.0}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
VLLM_RESTART_ON_CRASH="${VLLM_RESTART_ON_CRASH:-1}"
VLLM_RESTART_DELAY="${VLLM_RESTART_DELAY:-30}"
VLLM_MAX_RESTARTS="${VLLM_MAX_RESTARTS:-0}"

python - "${MIN_VLLM_VERSION}" <<'PY'
import importlib.metadata
import sys

from packaging.version import Version

minimum = Version(sys.argv[1])
try:
    current = Version(importlib.metadata.version("vllm"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit("vLLM is not installed. Install vllm>=0.17.0 before serving Qwen3.6.")

if current < minimum:
    raise SystemExit(
        f"vLLM {current} is too old for Qwen3.6; install vllm>={minimum}."
    )
PY

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

VLLM_ARGS=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
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

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

child_pid=""
stop_requested=0

stop_child() {
  stop_requested=1
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    log "stopping vLLM process ${child_pid}"
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
}

trap stop_child INT TERM

run_vllm_once() {
  "${VLLM_ARGS[@]}" &
  child_pid=$!
  log "started vLLM pid=${child_pid} model=${MODEL_PATH} served_name=${SERVED_MODEL_NAME} port=${PORT}"
  wait "${child_pid}"
  local status=$?
  child_pid=""
  return "${status}"
}

if [[ "${VLLM_RESTART_ON_CRASH}" != "1" ]]; then
  run_vllm_once
  exit $?
fi

restart_count=0
while true; do
  set +e
  run_vllm_once
  status=$?
  set -e

  if [[ "${stop_requested}" == "1" ]]; then
    log "vLLM launcher stopped"
    exit "${status}"
  fi

  if [[ "${status}" == "0" ]]; then
    log "vLLM exited cleanly; not restarting"
    exit 0
  fi

  restart_count=$((restart_count + 1))
  if [[ "${VLLM_MAX_RESTARTS}" != "0" && "${restart_count}" -gt "${VLLM_MAX_RESTARTS}" ]]; then
    log "vLLM crashed with status ${status}; max restarts (${VLLM_MAX_RESTARTS}) reached"
    exit "${status}"
  fi

  log "vLLM crashed with status ${status}; restarting in ${VLLM_RESTART_DELAY}s (attempt ${restart_count})"
  sleep "${VLLM_RESTART_DELAY}"
done
