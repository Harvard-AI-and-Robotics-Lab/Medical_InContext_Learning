# Remote 8GPU Transfer And Runbook

This document is for moving the clean final project from:

```text
mindazhao@10.100.241.227:/shared/ssd_14T/home/mindazhao/Incontext_Learning_Git
```

to:

```text
mindazhao@10.100.241.77
```

The immediate remote run target is:

1. BreaKHis benign/malignant binary classification with strict patient-level split.
2. DDR diabetic-retinopathy grading zero-shot only.
3. Run Qwen3.6 and Gemma4 concurrently through vLLM.
4. Use GPU UUIDs, not visible GPU ordinals, because CUDA/NVIDIA device ordering can be misleading.

## What Is In This Project

The repository contains:

- `src/`: dataset loaders, prompt contracts, retrieval, inference, metrics, encoders.
- `configs/final/`: aligned final YAML configs.
- `scripts/`: dataset preparation, feature extraction, vLLM drivers, VLM inference, ResNet50 training.
- `manifests/`: split manifests and fixed exemplar JSON files.
- `docs/`: runbooks and protocol documentation.
- `results/`: paper-ready LAG summary figures/tables.

The transfer package intentionally excludes heavy or machine-specific folders:

- `data/`
- `outputs/`
- `logs/`
- Python caches
- model cache folders

Datasets and features should be regenerated on the remote machine.

## Formal Settings Used Here

All VLM runs use the aligned final protocol:

```yaml
client_backend: vllm
temperature: 1.0
top_p: null
max_tokens: 512
response_format: json_object
enable_thinking: false
seed: 3407
batch_size: 256
parallel_requests: 256
k: 6
```

For BreaKHis binary:

- Labels: `benign`, `malignant`
- JSON output: `label`, `confidence`, scalar `probability`, `evidence`
- `probability` means `P(malignant)`
- Split: strict patient-level, 70/15/15 target, `seed=3407`
- Current generated split:
  - train/val/test images: 5507/1252/1150
  - train/val/test strict patients: 49/10/11
  - patient leakage: 0

For DDR zero-shot:

- Labels: `no_dr`, `mild_npdr`, `moderate_npdr`, `severe_npdr`, `proliferative_dr`, `ungradable`
- JSON output: `label`, `confidence`, class-wise `probabilities`, `evidence`
- Uses official DDR train/valid/test split.
- Preprocessing: crop fundus black border, preserve aspect ratio, resize long side to 1024, black-pad to 1024 x 1024.

## Transfer Package

On `10.100.241.227`, the package is created as:

```bash
/shared/ssd_14T/home/mindazhao/Incontext_Learning_Git_transfer_YYYYMMDD_HHMMSS.tar.gz
```

Copy it to the remote machine:

```bash
scp /shared/ssd_14T/home/mindazhao/Incontext_Learning_Git_transfer_YYYYMMDD_HHMMSS.tar.gz \
  mindazhao@10.100.241.77:/shared/ssd_14T/home/mindazhao/
```

On `10.100.241.77`:

```bash
cd /shared/ssd_14T/home/mindazhao
tar -xzf Incontext_Learning_Git_transfer_YYYYMMDD_HHMMSS.tar.gz
cd Incontext_Learning_Git
```

## Environment Setup

Use the existing environment if the remote machine already has vLLM and the required libraries. Otherwise:

```bash
conda create -n qwen36_vllm python=3.11 -y
conda activate qwen36_vllm
pip install -r requirements.txt
pip install vllm
```

If gated Hugging Face models are used, log in:

```bash
huggingface-cli login
```

Expected model variables:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3.6-27B
export GEMMA_MODEL_PATH=/path/to/google_gemma-4-31B-it
```

`QWEN_MODEL_PATH` may also be a Hugging Face model ID if the remote machine can download it.

## Dataset Download

### BreaKHis

```bash
mkdir -p data/raw
curl -L -o data/raw/BreaKHis_v1.tar.gz \
  http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz

python scripts/prepare_breakhis_patient_split.py \
  --archive data/raw/BreaKHis_v1.tar.gz \
  --extract-dir data/raw/BreaKHis_v1_extracted \
  --manifest-csv manifests/breakhis_patient_split_seed3407.csv \
  --manifest-json data/breakhis/manifest.json \
  --seed 3407 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --split-mode patient_multilabel_stratified_best_effort

python scripts/prepare_breakhis_binary_manifest.py
```

The second command converts the subtype manifest into binary benign/malignant labels while preserving the same patient-level split.

### DDR

The DDR Google Drive folder is:

```text
https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC
```

Use `rclone`, `gdown`, or manual download. The final extracted structure must contain:

```text
data/raw/DDR_extracted/DDR-dataset/DR_grading/train.txt
data/raw/DDR_extracted/DDR-dataset/DR_grading/valid.txt
data/raw/DDR_extracted/DDR-dataset/DR_grading/test.txt
data/raw/DDR_extracted/DDR-dataset/DR_grading/train/*.jpg
data/raw/DDR_extracted/DDR-dataset/DR_grading/valid/*.jpg
data/raw/DDR_extracted/DDR-dataset/DR_grading/test/*.jpg
```

Then build the manifest:

```bash
python scripts/prepare_ddr_manifest.py \
  --data-root data/raw/DDR_extracted/DDR-dataset \
  --manifest-csv manifests/ddr_official_split.csv \
  --manifest-json data/ddr/manifest.json \
  --fixed-exemplars-json manifests/ddr_fixed_exemplars_seed3407.json \
  --seed 3407
```

The remote driver will generate the 1024 crop-pad manifest automatically:

```text
manifests/ddr_official_split_crop_pad_1024.csv
```

## Choose GPU UUIDs

On the remote machine:

```bash
nvidia-smi -L
```

Pick two GPUs for Qwen3.6, two GPUs for Gemma4, optionally one GPU for feature extraction and one for ResNet50.

Example:

```bash
export QWEN_CUDA_VISIBLE_DEVICES=GPU-aaaa,GPU-bbbb
export GEMMA_CUDA_VISIBLE_DEVICES=GPU-cccc,GPU-dddd
export FEATURE_CUDA_VISIBLE_DEVICES=GPU-eeee
export RESNET_CUDA_VISIBLE_DEVICES=GPU-ffff
```

Use actual UUIDs from `nvidia-smi -L`.

## Smoke Test

Before launching the full run:

```bash
export LIMIT=2
export QWEN_MODEL_PATH=/path/to/Qwen3.6-27B
export GEMMA_MODEL_PATH=/path/to/google_gemma-4-31B-it
export QWEN_CUDA_VISIBLE_DEVICES=GPU-aaaa,GPU-bbbb
export GEMMA_CUDA_VISIBLE_DEVICES=GPU-cccc,GPU-dddd
export FEATURE_CUDA_VISIBLE_DEVICES=GPU-eeee
export RESNET_CUDA_VISIBLE_DEVICES=GPU-ffff

bash scripts/run_remote_breakhis_binary_ddr_zeroshot_driver.sh
unset LIMIT
```

Check logs:

```bash
tail -n 80 logs/remote_breakhis_binary_ddr_*/driver.log
tail -n 80 logs/remote_breakhis_binary_ddr_*/breakhis_binary_qwen_suite.log
tail -n 80 logs/remote_breakhis_binary_ddr_*/ddr_qwen_zero_shot.log
```

## Full Remote Run

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3.6-27B
export GEMMA_MODEL_PATH=/path/to/google_gemma-4-31B-it
export QWEN_CUDA_VISIBLE_DEVICES=GPU-aaaa,GPU-bbbb
export GEMMA_CUDA_VISIBLE_DEVICES=GPU-cccc,GPU-dddd
export FEATURE_CUDA_VISIBLE_DEVICES=GPU-eeee
export RESNET_CUDA_VISIBLE_DEVICES=GPU-ffff
export KEEP_VLLM=1

tmux new-session -d -s breakhis_binary_ddr \
  "cd /shared/ssd_14T/home/mindazhao/Incontext_Learning_Git && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate qwen36_vllm && \
   bash scripts/run_remote_breakhis_binary_ddr_zeroshot_driver.sh"
```

Monitor:

```bash
tmux capture-pane -t breakhis_binary_ddr -p -S -120
tail -f logs/remote_breakhis_binary_ddr_*/driver.log
```

## Outputs To Check

BreaKHis binary VLM:

```text
outputs/final/breakhis_binary_qwen36_noicl_fixed6_random6/breakhis_binary/{zero_shot,fixed_random_6,random_icl_k6}/
outputs/final/breakhis_binary_qwen36_clip_top6/breakhis_binary/rg_icl_global_similarity/
outputs/final/breakhis_binary_qwen36_dinov3_cls_top6/breakhis_binary/rg_icl_global_similarity/
outputs/final/breakhis_binary_qwen36_clip_dinov3cls_top6/breakhis_binary/rg_icl_global_similarity/

outputs/final/breakhis_binary_gemma4_noicl_fixed6_random6/breakhis_binary/{zero_shot,fixed_random_6,random_icl_k6}/
outputs/final/breakhis_binary_gemma4_clip_top6/breakhis_binary/rg_icl_global_similarity/
outputs/final/breakhis_binary_gemma4_dinov3_cls_top6/breakhis_binary/rg_icl_global_similarity/
outputs/final/breakhis_binary_gemma4_clip_dinov3cls_top6/breakhis_binary/rg_icl_global_similarity/
```

BreaKHis binary supervised:

```text
outputs/final/resnet50_breakhis_binary_384_seed3407/metrics.json
```

DDR zero-shot:

```text
outputs/final/ddr_qwen36_zero_shot_1024/ddr/zero_shot/
outputs/final/ddr_gemma4_zero_shot_1024/ddr/zero_shot/
```

Each method directory should contain:

```text
predictions.json
raw_outputs.jsonl
metrics.json
```

## Important Sanity Checks

1. BreaKHis binary manifest must have no patient leakage:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("manifests/breakhis_binary_patient_split_seed3407.csv", dtype={"patient_id": str})
print(pd.crosstab(df["split"], df["label_name"]))
leak = df.groupby("patient_id")["split"].nunique()
print("patient leakage n =", int((leak > 1).sum()))
PY
```

2. DDR processed images must be 1024 x 1024:

```bash
python - <<'PY'
from PIL import Image
import pandas as pd
df = pd.read_csv("manifests/ddr_official_split_crop_pad_1024.csv")
for p in df["image_path"].head(10):
    print(p, Image.open(p).size)
PY
```

3. Metrics must be full-test metrics, not smoke-test metrics. Confirm `n` equals:

- BreaKHis binary test: 1150 images
- DDR test: 4105 images

4. For DDR, old partial/native results can be misleading. Do not compare against a `metrics.json` with `n=2` or partial predictions.
