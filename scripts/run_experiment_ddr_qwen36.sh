#!/usr/bin/env bash

# source activate base
# conda activate medical_vlm_icl


python scripts/run_final_classification.py --config configs/final/ddr_qwen36_noicl_fixed6_random6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_dinov3_cls_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_dinov3cls_top6_512.yaml

python scripts/run_final_classification.py --config configs/final/ddr_gemma4_noicl_fixed6_random6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_gemma4_dinov3_cls_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_dinov3cls_top6_512.yaml