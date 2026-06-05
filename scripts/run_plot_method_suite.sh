#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${1:-lag}"
MODEL_GROUP="${2:-all}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run_plot_method_suite.sh <dataset> [qwen|gemma|vlm|lora|all]

Datasets:
  lag             Main LAG glaucoma configs.
  tbx11k          TBX11K three-class configs.
  ddr_512         DDR crop_pad_512 configs.
  breakhis_binary Pooled BreaKHis binary configs.

This script expands the plotted method rows explicitly. The old 8-config VLM
suite hid four plotted rows inside the two noicl_fixed6_random6 configs:
Qwen Fixed-6, Qwen Random-6, Gemma Fixed-6, and Gemma Random-6. This driver
runs those rows separately and optionally runs/evaluates the Gemma4 LoRA row.
EOF
}

case "${DATASET}" in
  lag)
    qwen_prefix="qwen36"
    gemma_prefix="gemma4"
    suffix=""
    lora_dataset="lag_project"
    lora_output_name="lag_project"
    ;;
  tbx11k)
    qwen_prefix="tbx11k_qwen36"
    gemma_prefix="tbx11k_gemma4"
    suffix=""
    lora_dataset="tbx11k"
    lora_output_name="tbx11k"
    ;;
  ddr_512)
    qwen_prefix="ddr_qwen36"
    gemma_prefix="ddr_gemma4"
    suffix="_512"
    lora_dataset="ddr_512"
    lora_output_name="ddr512"
    ;;
  breakhis_binary)
    qwen_prefix="breakhis_binary_qwen36"
    gemma_prefix="breakhis_binary_gemma4"
    suffix=""
    lora_dataset="breakhis_binary"
    lora_output_name="breakhis_binary"
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    usage
    exit 2
    ;;
esac

run_vlm_model() {
  local model_name="$1"
  local prefix="$2"

  local noicl_cfg="configs/final/${prefix}_noicl_fixed6_random6${suffix}.yaml"
  local clip_cfg="configs/final/${prefix}_clip_top6${suffix}.yaml"
  local dino_cfg="configs/final/${prefix}_dinov3_cls_top6${suffix}.yaml"
  local fusion_cfg="configs/final/${prefix}_clip_dinov3cls_top6${suffix}.yaml"

  echo "[${model_name}] zero_shot"
  python scripts/run_final_classification.py --config "${noicl_cfg}" --methods zero_shot "${LIMIT_ARGS[@]}"

  echo "[${model_name}] fixed_random_6"
  python scripts/run_final_classification.py --config "${noicl_cfg}" --methods fixed_random_6 "${LIMIT_ARGS[@]}"

  echo "[${model_name}] random_icl"
  python scripts/run_final_classification.py --config "${noicl_cfg}" --methods random_icl "${LIMIT_ARGS[@]}"

  echo "[${model_name}] clip_top6"
  python scripts/run_final_classification.py --config "${clip_cfg}" "${LIMIT_ARGS[@]}"

  echo "[${model_name}] dinov3_cls_top6"
  python scripts/run_final_classification.py --config "${dino_cfg}" "${LIMIT_ARGS[@]}"

  echo "[${model_name}] clip_dinov3cls_top6"
  python scripts/run_final_classification.py --config "${fusion_cfg}" "${LIMIT_ARGS[@]}"
}

run_lora() {
  local out_dir="${LORA_OUTPUT_DIR:-outputs/lora/gemma4_language_lora_${lora_output_name}_r16_a32_lr1e-4_seed3407}"
  local adapter_dir="${LORA_ADAPTER_DIR:-${out_dir}/best}"

  if [[ "${TRAIN_LORA:-1}" == "1" && ! -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "[lora] train ${lora_dataset}"
    CUDA_VISIBLE_DEVICES="${LORA_CUDA_VISIBLE_DEVICES:-0,1}" \
      python scripts/train_gemma4_language_lora.py \
        --dataset "${lora_dataset}" \
        --output_dir "${out_dir}" \
        --epochs "${LORA_EPOCHS:-3}" \
        --batch_size "${LORA_BATCH_SIZE:-1}" \
        --grad_accum "${LORA_GRAD_ACCUM:-16}" \
        --lr "${LORA_LR:-1e-4}" \
        --lora_r "${LORA_R:-16}" \
        --lora_alpha "${LORA_ALPHA:-32}" \
        --lora_dropout "${LORA_DROPOUT:-0.05}" \
        --eval_steps "${LORA_EVAL_STEPS:-200}" \
        --eval_subset "${LORA_EVAL_SUBSET:-200}"
  fi

  echo "[lora] generation test ${lora_dataset}"
  CUDA_VISIBLE_DEVICES="${LORA_CUDA_VISIBLE_DEVICES:-0,1}" \
    python scripts/eval_gemma4_language_lora.py \
      --dataset "${lora_dataset}" \
      --adapter_dir "${adapter_dir}" \
      --output_json "${out_dir}/test_generation_metrics.json" \
      --predictions_json "${out_dir}/test_generation_predictions.json" \
      --split test \
      --mode generate \
      --max_new_tokens "${LORA_MAX_NEW_TOKENS:-512}"

  if [[ "${LORA_LOGPROB_AUC:-0}" == "1" ]]; then
    echo "[lora] logprob test ${lora_dataset}"
    CUDA_VISIBLE_DEVICES="${LORA_CUDA_VISIBLE_DEVICES:-0,1}" \
      python scripts/eval_gemma4_language_lora.py \
        --dataset "${lora_dataset}" \
        --adapter_dir "${adapter_dir}" \
        --output_json "${out_dir}/test_logprob_metrics.json" \
        --predictions_json "${out_dir}/test_logprob_predictions.json" \
        --split test \
        --mode logprob
  fi
}

case "${MODEL_GROUP}" in
  qwen)
    run_vlm_model qwen36 "${qwen_prefix}"
    ;;
  gemma)
    run_vlm_model gemma4 "${gemma_prefix}"
    ;;
  vlm)
    run_vlm_model qwen36 "${qwen_prefix}"
    run_vlm_model gemma4 "${gemma_prefix}"
    ;;
  lora)
    run_lora
    ;;
  all)
    run_vlm_model qwen36 "${qwen_prefix}"
    run_vlm_model gemma4 "${gemma_prefix}"
    run_lora
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown model group: ${MODEL_GROUP}" >&2
    usage
    exit 2
    ;;
esac
