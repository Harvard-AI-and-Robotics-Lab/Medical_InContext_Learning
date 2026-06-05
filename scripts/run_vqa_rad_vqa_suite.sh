#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

MODEL_GROUP="${1:-all}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

wait_for_features() {
  local required=(
    outputs/features_clip_global_vqa_rad_refval_promptfix/vqa_rad/clip/metadata.json
    outputs/features_dinov3_global_vqa_rad_refval_promptfix/vqa_rad/dinov3/metadata.json
    outputs/features_clip_dinov3cls_05_global_vqa_rad_refval_promptfix/vqa_rad/clip_dinov3cls05/metadata.json
  )
  local path
  while true; do
    local ready=1
    for path in "${required[@]}"; do
      if [[ ! -s "${path}" ]]; then
        ready=0
        break
      fi
    done
    if [[ "${ready}" == "1" ]]; then
      return 0
    fi
    echo "[wait] VQA-RAD retrieval features are not ready; sleeping 30s"
    sleep 30
  done
}

run_model() {
  local prefix="$1"
  python3 scripts/run_final_vqa.py --config "configs/final/vqa_rad_${prefix}_noicl_fixed6_random6.yaml" "${LIMIT_ARGS[@]}"
  wait_for_features
  python3 scripts/run_final_vqa.py --config "configs/final/vqa_rad_${prefix}_clip_top6.yaml" "${LIMIT_ARGS[@]}"
  python3 scripts/run_final_vqa.py --config "configs/final/vqa_rad_${prefix}_dinov3_cls_top6.yaml" "${LIMIT_ARGS[@]}"
  python3 scripts/run_final_vqa.py --config "configs/final/vqa_rad_${prefix}_clip_dinov3cls_top6.yaml" "${LIMIT_ARGS[@]}"
}

case "${MODEL_GROUP}" in
  qwen|qwen36)
    run_model qwen36
    ;;
  gemma|gemma4)
    run_model gemma4
    ;;
  medgemma|medgemma27b)
    run_model medgemma27b
    ;;
  all)
    mkdir -p logs/vqa_rad_vqa
    tmux new-session -d -s vqa_rad_qwen36_eval "bash -lc 'cd \"$(pwd)\" && LIMIT=\"${LIMIT:-}\" scripts/run_vqa_rad_vqa_suite.sh qwen36' > logs/vqa_rad_vqa/qwen36_eval.log 2>&1"
    tmux new-session -d -s vqa_rad_gemma4_eval "bash -lc 'cd \"$(pwd)\" && LIMIT=\"${LIMIT:-}\" scripts/run_vqa_rad_vqa_suite.sh gemma4' > logs/vqa_rad_vqa/gemma4_eval.log 2>&1"
    tmux new-session -d -s vqa_rad_medgemma27b_eval "bash -lc 'cd \"$(pwd)\" && LIMIT=\"${LIMIT:-}\" scripts/run_vqa_rad_vqa_suite.sh medgemma27b' > logs/vqa_rad_vqa/medgemma27b_eval.log 2>&1"
    ;;
  *)
    echo "Usage: scripts/run_vqa_rad_vqa_suite.sh [qwen36|gemma4|medgemma27b|all]" >&2
    exit 2
    ;;
esac
