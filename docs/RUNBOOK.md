# Runbook

## 1. Validate Splits and Configs

The final configs expect the LAG image root at `data/LAG`. Create a local
symlink or edit `data_root` in the configs:

```bash
mkdir -p data
ln -s /path/to/LAG/image/root data/LAG
```

```bash
python scripts/validate_final_setup.py
```

This checks:

- required manifest columns exist;
- train, val, and test IDs do not overlap;
- fixed exemplars are train-only;
- final VLM configs use the aligned generation protocol.

## 2. Start vLLM Servers

Use GPU UUIDs on machines where CUDA ordinal mapping is unreliable:

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID_1,GPU_UUID_2 bash scripts/start_qwen36_vllm.sh
CUDA_VISIBLE_DEVICES=GPU_UUID_3,GPU_UUID_4 MODEL_PATH=/path/to/gemma4-31b-it bash scripts/start_gemma4_vllm.sh
```

Default served model names:

- Qwen3.6: `qwen36-27b-final` on `http://127.0.0.1:18001/v1`
- Gemma4: `gemma4-31b-final` on `http://127.0.0.1:18002/v1`

## 3. Build Retrieval Features

```bash
python scripts/extract_features.py --config configs/final/extract_clip_global.yaml
python scripts/extract_features.py --config configs/final/extract_dinov3_global.yaml

python scripts/build_fused_features.py \
  --feature-a outputs/features_clip_global/lag_project/clip \
  --feature-b outputs/features_dinov3_global/lag_project/dinov3 \
  --output-dir outputs/features_clip_dinov3cls_05_global/lag_project/clip_dinov3cls05 \
  --encoder-name clip_dinov3cls05 \
  --score-definition "0.5*CLIP_global_cosine + 0.5*DINOv3_CLS_cosine"
```

## 4. Run Final VLM Suite

```bash
bash scripts/run_final_suite.sh
```

Or run individual configs:

```bash
python scripts/run_final_classification.py --config configs/final/qwen36_clip_dinov3cls_top6.yaml
python scripts/run_final_classification.py --config configs/final/gemma4_clip_dinov3cls_top6.yaml
```

## 5. Run ResNet50 Baseline

```bash
python scripts/train_resnet50_lag.py \
  --manifest-csv manifests/lag_manifest.csv \
  --data-root data/LAG \
  --output-dir outputs/final/resnet50_lag_512_seed3407 \
  --image-size 512 \
  --batch-size 32 \
  --epochs 40 \
  --patience 8 \
  --num-workers 8 \
  --seed 3407 \
  --backbone-lr 3e-5 \
  --head-lr 1e-3 \
  --weight-decay 1e-4
```
