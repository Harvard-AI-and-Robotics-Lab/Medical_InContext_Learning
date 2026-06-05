#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_GROUP="${1:-all}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

QWEN_CONFIGS=(
  configs/final/breakhis_binary_qwen36_noicl_fixed6_random6.yaml
  configs/final/breakhis_binary_qwen36_clip_top6.yaml
  configs/final/breakhis_binary_qwen36_dinov3_cls_top6.yaml
  configs/final/breakhis_binary_qwen36_clip_dinov3cls_top6.yaml
)

GEMMA_CONFIGS=(
  configs/final/breakhis_binary_gemma4_noicl_fixed6_random6.yaml
  configs/final/breakhis_binary_gemma4_clip_top6.yaml
  configs/final/breakhis_binary_gemma4_dinov3_cls_top6.yaml
  configs/final/breakhis_binary_gemma4_clip_dinov3cls_top6.yaml
)

run_configs() {
  local -n configs_ref=$1
  for cfg in "${configs_ref[@]}"; do
    echo "Running ${cfg}"
    python scripts/run_final_classification.py --config "${cfg}" "${LIMIT_ARGS[@]}"
  done
}

case "${MODEL_GROUP}" in
  qwen)
    run_configs QWEN_CONFIGS
    ;;
  gemma)
    run_configs GEMMA_CONFIGS
    ;;
  all)
    run_configs QWEN_CONFIGS
    run_configs GEMMA_CONFIGS
    ;;
  *)
    echo "Usage: $0 [qwen|gemma|all]" >&2
    exit 2
    ;;
esac
