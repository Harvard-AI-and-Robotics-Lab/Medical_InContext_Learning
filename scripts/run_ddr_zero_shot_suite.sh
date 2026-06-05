#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_GROUP="${1:-all}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

case "${MODEL_GROUP}" in
  qwen)
    python scripts/run_final_classification.py --config configs/final/ddr_qwen36_zero_shot.yaml "${LIMIT_ARGS[@]}"
    ;;
  gemma)
    python scripts/run_final_classification.py --config configs/final/ddr_gemma4_zero_shot.yaml "${LIMIT_ARGS[@]}"
    ;;
  all)
    python scripts/run_final_classification.py --config configs/final/ddr_qwen36_zero_shot.yaml "${LIMIT_ARGS[@]}"
    python scripts/run_final_classification.py --config configs/final/ddr_gemma4_zero_shot.yaml "${LIMIT_ARGS[@]}"
    ;;
  *)
    echo "Usage: $0 [qwen|gemma|all]" >&2
    exit 2
    ;;
esac
