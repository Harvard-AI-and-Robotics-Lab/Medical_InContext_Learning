# Retrieval-Guided Medical VLM In-Context Learning

This repository contains the reproducible code path for the final medical image
classification and medical visual question answering experiments. It is
intended to let another researcher prepare the same datasets, launch the same
vLLM servers, run the same VLM/VQA main experiments, run the same LLM-judge
evaluation, and train the same supervised baselines without changing the
experimental protocol.

Raw datasets are intentionally not included. The repository includes source
code, final configs, split manifests, fixed exemplar files, preprocessing
scripts, feature extraction scripts, VLM/VQA inference entrypoints, ResNet
training entrypoints, and protocol documentation.

## Experimental Contract

All final VLM experiments use the same six-method matrix for each VLM:

| Method | Config method name | Reference source |
| --- | --- | --- |
| Zero-shot | `zero_shot` | none |
| Fixed exemplar | `fixed_random_6` | fixed six train examples from `manifests/*fixed_exemplars*.json` |
| Random exemplar | `random_icl` | deterministic per-query train-split sample |
| CLIP top-6 | `rg_icl_global_similarity_k6` | `openai/clip-vit-large-patch14` global embedding |
| DINOv3 top-6 | `rg_icl_global_similarity_k6` | DINOv3 CLS/global embedding |
| CLIP0.5 + DINOv3 0.5 top-6 | `rg_icl_dual_global_similarity_k6` | equal-weight fused normalized CLIP and DINOv3 embeddings |

The train split is the only allowed reference pool. Validation and test images
must never be used as ICL references.

Paper plots expand the compact config suite into explicit plotted rows. The
eight Qwen3.6/Gemma4 VLM configs correspond to twelve VLM plot rows because the
`*_noicl_fixed6_random6.yaml` configs each contain `zero_shot`,
`fixed_random_6`, and `random_icl`. When the Gemma4 language-LoRA baseline is
included, there are thirteen non-supervised rows, with ViT and ResNet50 shown as
separate supervised baselines. See `docs/PLOT_METHOD_ROWS.md` and
`scripts/run_plot_method_suite.sh` for the exact one-to-one mapping. The mapping
can be smoke-tested without launching models via
`python scripts/validate_plot_method_suite.py`.

Final VLM protocol:

```yaml
client_backend: vllm
temperature: 1.0
top_p: null
response_format: json_object
enable_thinking: false
seed: 3407
k: 6
```

The two VLMs used by the final configs are:

| Model alias in configs | Intended model |
| --- | --- |
| `qwen36-27b-final` | `Qwen/Qwen3.6-27B` |
| `gemma4-31b-final` | `google/gemma-4-31B-it` |

VQA experiments use the same six-method structure and add MedGemma:

| Model alias in VQA configs | Intended model |
| --- | --- |
| `qwen36-27b-slake`, `qwen36-27b-pathvqa`, `qwen36-27b-vqa-rad`, `qwen36-27b-vqamed2019` | `Qwen/Qwen3.6-27B` |
| `gemma4-31b-slake`, `gemma4-31b-pathvqa`, `gemma4-31b-vqa-rad`, `gemma4-31b-vqamed2019` | `google/gemma-4-31B-it` |
| `medgemma-27b-slake`, `medgemma-27b-pathvqa`, `medgemma-27b-vqa-rad`, `medgemma-27b-vqamed2019` | `google/medgemma-27b-it` |

Final VQA generation settings:

```yaml
client_backend: vllm
temperature: 1.0
top_p: null
max_tokens: 512
response_format: json_object
enable_thinking: false
seed: 3407
k: 6
image_max_side: null
```

VQA answers are evaluated with the LLM-judge prompt in
`configs/judge/vqa_llm_judge_prompt_v1.txt`. The prompt must remain byte-for-byte
unchanged for the reported VQA judge scores:

```text
SHA256 20d61c55dd8a5db11cce4fa3cfa5b13bcdf32eede2b3c447132c946375caba65
model  gpt-5.4-mini-2026-03-17
temperature 0.0
```

## Final Dataset Settings

### BreaKHis

Task: benign/malignant binary classification.

Required final setting:

- strict patient-level split;
- seed `3407`;
- run separately by magnification: `40X`, `100X`, `200X`, `400X`;
- fixed exemplars are sampled within each magnification-specific train split;
- random exemplars are sampled within each magnification-specific train split;
- do not force `3 benign + 3 malignant`;
- VLM output includes binary label, malignant probability, histology subtype,
  confidence, and short evidence.

Manifests:

```text
manifests/breakhis_binary_patient_split_seed3407_mag40.csv
manifests/breakhis_binary_patient_split_seed3407_mag100.csv
manifests/breakhis_binary_patient_split_seed3407_mag200.csv
manifests/breakhis_binary_patient_split_seed3407_mag400.csv
manifests/breakhis_binary_fixed_exemplars_seed3407_mag40.json
manifests/breakhis_binary_fixed_exemplars_seed3407_mag100.json
manifests/breakhis_binary_fixed_exemplars_seed3407_mag200.json
manifests/breakhis_binary_fixed_exemplars_seed3407_mag400.json
```

ResNet50 supervised baseline:

- BreaKHis 8-class subtype classification;
- ImageNet-pretrained ResNet50;
- image size `384`;
- epochs `40`;
- patience `8`;
- validation selection metric: accuracy;
- pooled and magnification-specific runs are provided.

### DDR

Task: diabetic-retinopathy grading.

Required final setting:

- official DDR split;
- preprocess fundus images with crop/pad and exact `512x512` output for VLM
  main experiments;
- six VLM main experiments for Qwen3.6 and Gemma4;
- ResNet50 baseline uses `image_size=384` on the `crop_pad_512` image set.

Main manifest:

```text
manifests/ddr_official_split_crop_pad_512.csv
```

The legacy `crop_pad_1024` manifest is retained for audit/history, but the
final runnable DDR ResNet setting uses `crop_pad_512` because that image set is
the complete generated set in this workflow.

### TBX11K

Task: three-class chest X-ray classification.

The final TBX11K label set is:

```text
healthy
sick but non-TB
TB
```

Required final setting:

- use `manifests/tbx11k_train85_val15_officialval_test_seed3407.csv`;
- split the official TBX11K train list into train/validation with seed `3407`;
- treat the official TBX11K validation list as the test set;
- fixed and random exemplars are sampled only from the train split;
- do not present the val-as-test result as the original hidden CVPR benchmark.

Manifests:

```text
manifests/tbx11k_train85_val15_officialval_test_seed3407.csv
manifests/tbx11k_train85_val15_officialval_test_seed3407.summary.json
manifests/tbx11k_fixed_exemplars_seed3407.json
```

Supervised baselines:

- ResNet50 uses ImageNet weights, `image_size=384`, epochs `40`, patience `8`;
- ViT-B/16 uses `google/vit-base-patch16-224`, native `224x224`, epochs `200`, patience `25`, minimum epochs `50`.

### CheXpert

Task: five-pathology CheXpert multi-label classification.

The only target labels for the final CheXpert experiments are:

```text
atelectasis
cardiomegaly
consolidation
edema
pleural_effusion
```

Required final setting:

- official train/validation/test split;
- test images correspond to the CheXlocalize/CheXpert test image set;
- use exact `320x320` resized images;
- prompt references and target schema list only the five labels above;
- do not use the full 14-label CheXpert target for final results.

Main manifest:

```text
manifests/chexpert_official_split_320.csv
```

ResNet50 supervised baseline:

- ImageNet-pretrained ResNet50;
- multi-label binary heads for the five final pathologies;
- image size `320`;
- epochs `20`;
- patience `5`.

### VQA-RAD

Task: radiology visual question answering.

Required final VQA setting:

- Hugging Face `flaviagiammarino/vqa-rad`;
- `scripts/prepare_vqa_rad_dataset.py` writes the current manifest-json layout
  under `data/vqa/vqa_rad`;
- original image size is used in the current VQA configs
  (`image_max_side: null`);
- training/reference and validation samples are used only as ICL reference
  candidates in the `manifest_refval` configs;
- the test split is never used as an ICL reference pool;
- six VQA methods for Qwen3.6, Gemma4, and MedGemma27B;
- raw model responses are saved for every sample before LLM-judge evaluation;
- LLM-judge uses the same GPT batch protocol as the other VQA datasets.

Current manifest and fixed exemplars:

```text
data/vqa/vqa_rad/manifest.json
data/vqa/vqa_rad/manifest_refval.json
data/vqa/vqa_rad/fixed_exemplars_refval_seed3407.json
```

Legacy 512 preprocessing support is retained for audit/history:

- start from the Hugging Face VQA-RAD export under `data/VQA_RAD`;
- build a CSV manifest with question, answer, split, and image path columns;
- resize all images to exact `512x512` JPEGs with bicubic interpolation and no
  cropping;
- extract retrieval features from the resized manifest with CLIP at `224x224`
  and DINOv3 at `512x512`;
- build equal-weight CLIP+DINOv3 fused global features.

Manifests:

```text
manifests/vqa_rad_official_split.csv
manifests/vqa_rad_official_split_512.csv
```

Processed image root:

```text
data/processed/vqa_rad_512
```

### SLAKE VQA

Task: medical visual question answering.

Required final setting:

- English-only SLAKE questions;
- original image size, no forced resize in the VQA configs;
- train split is used only as the ICL reference pool;
- validation and test images are never used as ICL references;
- six VQA methods for Qwen3.6, Gemma4, and MedGemma27B;
- raw model responses are saved for every sample before LLM-judge evaluation.

Main manifest and fixed exemplars:

```text
data/vqa/slake/manifest_en.json
data/vqa/slake/fixed_exemplars_en_seed3407.json
```

### PathVQA

Task: pathology visual question answering.

Required final setting:

- Hugging Face `flaviagiammarino/path-vqa`;
- original image size, no forced resize in the VQA configs;
- training split is used only as the ICL reference pool;
- six VQA methods for Qwen3.6, Gemma4, and MedGemma27B;
- raw model responses are saved for every sample before LLM-judge evaluation.

Main manifest and fixed exemplars:

```text
data/vqa/pathvqa/manifest.json
data/vqa/pathvqa/fixed_exemplars_seed3407.json
```

### VQA-Med2019

Task: ImageCLEF 2019 medical visual question answering.

Required final setting:

- official Zenodo VQA-Med2019 train, validation, and test files;
- original image size, no forced resize in the VQA configs;
- training split is used only as the ICL reference pool;
- validation/test images are never used as ICL references;
- six VQA methods for Qwen3.6, Gemma4, and MedGemma27B;
- raw model responses are saved for every sample before LLM-judge evaluation;
- LLM-judge can be run either directly or through the OpenAI Batch API wrapper.

Main manifest and fixed exemplars:

```text
data/vqa/vqamed2019/manifest.json
data/vqa/vqamed2019/fixed_exemplars_seed3407.json
```

### PathMMU

PathMMU support is included for feasibility and future VQA-style experiments.
The current public Hugging Face/GitHub release exposes `val`, `test`, and
`test_tiny` metadata, but not the paper's train split. The Hugging Face image
archive covers PubMed and EduContent images directly; SocialPath, Atlas, and
most PathCLS images require the additional official acquisition steps. Do not
report the HF-available subset as the full official PathMMU benchmark.

## Repository Layout

```text
configs/final/        Final VLM, feature, and ResNet configs
docs/                 Protocol notes, leakage checks, runbooks
manifests/            Split manifests and fixed exemplar files
scripts/              Data preparation, feature extraction, VLM, ResNet runners
src/                  Dataset, prompting, retrieval, inference, and metrics code
results/              Small curated result tables/figures only
outputs/              Runtime artifacts, ignored by git
logs/                 Runtime logs, ignored by git
```

## Environment

Create one Python environment with PyTorch, vLLM, Transformers, scikit-learn,
Pillow, pandas, NumPy, and the project package:

```bash
conda create -n medical_vlm_icl python=3.10 -y
conda activate medical_vlm_icl
pip install -r requirements.txt
pip install -e .
```

Install a CUDA/PyTorch/vLLM build compatible with the target cluster. The
provided configs assume vLLM-compatible local OpenAI-style endpoints.

## Data Placement

Raw data are not tracked. Create the expected local directories or symlinks:

```bash
mkdir -p data/raw data/processed
```

BreaKHis:

```bash
# Download externally:
# http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz

mkdir -p data/raw
tar -xzf /path/to/BreaKHis_v1.tar.gz -C data/raw/BreaKHis_v1_extracted
```

DDR:

```bash
# Place official DDR files/images so that manifests/ddr_official_split.csv
# image_path entries resolve relative to repo root.
```

CheXpert:

```bash
# Place CheXpert train/valid data and CheXlocalize test images so that
# manifests/chexpert_official_split.csv image_path entries resolve relative to
# repo root.
```

TBX11K:

```bash
# Place or extract TBX11K so manifest paths resolve, for example:
# data/raw/TBX11K_extracted/TBX11K/imgs/health/h0001.png
# data/raw/TBX11K_extracted/TBX11K/imgs/sick/s0001.png
# data/raw/TBX11K_extracted/TBX11K/imgs/tb/t0001.png
```

VQA-RAD:

```bash
# Current VQA workflow: download Hugging Face rows, save images, and build
# data/vqa/vqa_rad/manifest*.json files.
python scripts/prepare_vqa_rad_dataset.py

# Legacy 512 workflow: export Hugging Face parquet rows to data/VQA_RAD first.
python scripts/download_dataset_vqa_rad.py

# Expected exported files include:
# data/VQA_RAD/metadata.csv
# data/VQA_RAD/images/train/*.jpg
# data/VQA_RAD/images/test/*.jpg
```

VQA datasets:

```bash
python scripts/prepare_slake_dataset.py
python scripts/prepare_pathvqa_dataset.py
python scripts/prepare_vqamed2019_dataset.py
```

These scripts write only local data artifacts under `data/vqa/`, which is
ignored by git.

After data placement, verify manifest paths:

```bash
python scripts/validate_final_setup.py
```

## vLLM Servers

Start one server per model. GPU assignment can be changed, but config
`base_url` fields must match the launched ports.

```bash
export VLLM_API_KEY=EMPTY

CUDA_VISIBLE_DEVICES=0,1 MODEL_PATH=/path/to/Qwen3.6-27B \
  PORT=18003 GPU_MEMORY_UTILIZATION=0.9 SERVED_MODEL_NAME=qwen36-27b-final \
  bash scripts/start_qwen36_vllm.sh

CUDA_VISIBLE_DEVICES=2,3 MODEL_PATH=/path/to/gemma-4-31B-it \
  PORT=18005 GPU_MEMORY_UTILIZATION=0.9 SERVED_MODEL_NAME=gemma4-31b-final \
  bash scripts/start_gemma4_vllm.sh
```

If a config points to a different port, edit only `inference.base_url` in the
YAML file or launch the server on the configured port.

Final config ports used in this repository:

| Dataset/config group | Qwen3.6 port | Gemma4 port |
| --- | ---: | ---: |
| BreaKHis binary magnification configs | `18001` | `18002` |
| DDR 512 configs | `18003` | `18002` |
| CheXpert5 320 configs | `18003` | `18005` |

Potential Troubleshooting:

```bash
pip install --force-reinstall flashinfer-cubin==0.6.4 \
  --extra-index-url https://flashinfer.ai/whl/cu128 \
  'flashinfer-jit-cache==0.6.4+cu128'
```

VQA servers use three two-GPU vLLM endpoints. The provided helper scripts start
the configured aliases and ports:

```bash
bash scripts/start_slake_vllm_servers.sh
bash scripts/start_pathvqa_vllm_servers.sh
```

SLAKE ports are `18101` Qwen3.6, `18102` Gemma4, and `18103` MedGemma27B.
PathVQA ports are `18201` Qwen3.6, `18202` Gemma4, and `18203` MedGemma27B.
VQA-Med2019 ports are `18301` Qwen3.6, `18302` Gemma4, and `18303`
MedGemma27B. VQA-RAD ports are `18401` Qwen3.6, `18402` Gemma4, and `18403`
MedGemma27B.

## Feature Extraction

Run CLIP and DINOv3 before retrieval-guided ICL. The fused retrieval feature is
constructed from the two extracted feature sets with weights `0.5/0.5`.

DDR 512:

```bash
bash scripts/run_ddr_512_prepare_features.sh
```

CheXpert5 320 final features:

```bash
bash scripts/run_chexpert_fast_features.sh
```

`run_chexpert_fast_features.sh` produces the exact `_fast` feature directories used by the final `chexpert5_*` configs.

VQA-RAD 512 preprocessing and features:

```bash
bash scripts/run_vqa_rad_512_pipeline.sh
```

To generate only the CSV manifests and exact `512x512` images, without GPU
feature extraction:

```bash
SKIP_FEATURES=1 bash scripts/run_vqa_rad_512_pipeline.sh
```

The full pipeline writes:

```text
manifests/vqa_rad_official_split.csv
manifests/vqa_rad_official_split_512.csv
data/processed/vqa_rad_512
outputs/features_clip_global_vqa_rad_512/vqa_rad/clip
outputs/features_dinov3_global_vqa_rad_512/vqa_rad/dinov3
outputs/features_clip_dinov3cls_05_global_vqa_rad_512/vqa_rad/clip_dinov3cls05
```

TBX11K final features:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py --config configs/final/tbx11k_extract_clip_global.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/extract_features.py --config configs/final/tbx11k_extract_dinov3_global.yaml
python scripts/build_fused_features.py \
  --feature-a outputs/features_clip_global_tbx11k/tbx11k/clip \
  --feature-b outputs/features_dinov3_global_tbx11k/tbx11k/dinov3 \
  --output-dir outputs/features_clip_dinov3cls_05_global_tbx11k/tbx11k/clip_dinov3cls05 \
  --weight-a 0.5 \
  --weight-b 0.5 \
  --encoder-name clip_dinov3cls05
```

Alternatively, use the driver:

```bash
bash scripts/run_tbx11k_formal_driver.sh
```

BreaKHis binary by magnification:

```bash
bash scripts/run_breakhis_binary_feature_pipeline.sh
```

This generates the four magnification-specific feature roots referenced by the final `breakhis_binary_mag{40,100,200,400}_*` configs.

VQA features:

```bash
bash scripts/run_slake_feature_pipeline.sh
bash scripts/run_pathvqa_feature_pipeline.sh
bash scripts/run_vqamed2019_feature_pipeline.sh
bash scripts/run_vqa_rad_feature_pipeline.sh
```

These produce CLIP, DINOv3, and equal-weight fused CLIP+DINO feature roots used
by the VQA retrieval configs.

## VLM Main Experiments

Each `*_noicl_fixed6_random6*.yaml` config runs three methods: `zero_shot`,
`fixed_random_6`, and `random_icl`. The CLIP, DINOv3, and fused configs each
run one retrieval method. Together these are the six main VLM experiments.

### DDR 512

```bash
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_noicl_fixed6_random6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_dinov3_cls_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_dinov3cls_top6_512.yaml

python scripts/run_final_classification.py --config configs/final/ddr_gemma4_noicl_fixed6_random6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_gemma4_dinov3_cls_top6_512.yaml
python scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_dinov3cls_top6_512.yaml
```

### CheXpert5 320

Use the `chexpert5_*` configs for final five-label results:

```bash
python scripts/run_final_classification.py --config configs/final/chexpert5_qwen36_noicl_fixed6_random6_320.yaml
python scripts/run_final_classification.py --config configs/final/chexpert5_qwen36_clip_top6_320.yaml
python scripts/run_final_classification.py --config configs/final/chexpert5_qwen36_dinov3_cls_top6_320.yaml
python scripts/run_final_classification.py --config configs/final/chexpert5_qwen36_clip_dinov3cls_top6_320.yaml

python scripts/run_final_classification.py --config configs/final/chexpert5_gemma4_noicl_fixed6_random6_320.yaml
python scripts/run_final_classification.py --config configs/final/chexpert5_gemma4_clip_top6_320.yaml
python scripts/run_final_classification.py --config configs/final/chexpert5_gemma4_dinov3_cls_top6_320.yaml
python scripts/run_final_classification.py --config configs/final/chexpert5_gemma4_clip_dinov3cls_top6_320.yaml
```

### TBX11K

```bash
python scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_noicl_fixed6_random6.yaml
python scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_clip_top6.yaml
python scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_dinov3_cls_top6.yaml
python scripts/run_final_classification.py --config configs/final/tbx11k_qwen36_clip_dinov3cls_top6.yaml

python scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_noicl_fixed6_random6.yaml
python scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_clip_top6.yaml
python scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_dinov3_cls_top6.yaml
python scripts/run_final_classification.py --config configs/final/tbx11k_gemma4_clip_dinov3cls_top6.yaml
```

### VQA-RAD

Prepare current VQA-RAD manifests and retrieval features first:

```bash
python scripts/prepare_vqa_rad_dataset.py
bash scripts/run_vqa_rad_feature_pipeline.sh
```

Then run the current VQA-RAD VQA config matrix with the unified VQA runner:

```bash
bash scripts/run_vqa_rad_vqa_suite.sh all
```

The VQA-RAD retrieval prompts use the same score-visible protocol as the other
VQA datasets: each retrieved reference includes its CLIP, DINOv3, or fused
similarity score in the target-LLM prompt and records the scores in
`prompt_metadata.neighbor_scores`.

The legacy exact-512 image workflow remains available through
`scripts/run_vqa_rad_512_pipeline.sh` and the `configs/final/vqa_rad_*_512.yaml`
configs, but the current VQA main configs use `data/vqa/vqa_rad/*.json`
manifests and original image size.

### BreaKHis Binary By Magnification

Run the four magnification blocks. Example for `40X`; repeat for `100`, `200`,
and `400` by changing `mag40` in the config names:

```bash
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_qwen36_noicl_fixed6_random6_subtype.yaml
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_qwen36_clip_top6_subtype.yaml
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_qwen36_dinov3_cls_top6_subtype.yaml
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_qwen36_clip_dinov3cls05_top6_subtype.yaml

python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_gemma4_noicl_fixed6_random6_subtype.yaml
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_gemma4_clip_top6_subtype.yaml
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_gemma4_dinov3_cls_top6_subtype.yaml
python scripts/run_final_classification.py --config configs/final/breakhis_binary_mag40_gemma4_clip_dinov3cls05_top6_subtype.yaml
```

## VQA Main Experiments

Each dataset has 18 main VQA runs: three models times six methods. The helper
scripts run the same config matrix used for the reported experiments:

```bash
bash scripts/run_slake_vqa_suite.sh all
bash scripts/run_pathvqa_vqa_suite.sh all
bash scripts/run_vqamed2019_vqa_retrieval_suite.sh all
bash scripts/run_vqa_rad_vqa_suite.sh all
```

To run one model group:

```bash
bash scripts/run_slake_vqa_suite.sh qwen36
bash scripts/run_slake_vqa_suite.sh gemma4
bash scripts/run_slake_vqa_suite.sh medgemma27b
```

The same model arguments are accepted by `scripts/run_pathvqa_vqa_suite.sh`
`scripts/run_vqamed2019_vqa_retrieval_suite.sh`, and
`scripts/run_vqa_rad_vqa_suite.sh`.

## VQA LLM-Judge Evaluation

The judge prompt and judge config are:

```text
configs/judge/vqa_llm_judge_prompt_v1.txt
configs/judge/vqa_llm_judge_gpt54mini.yaml
```

Use `gpt-5.4-mini-2026-03-17` at temperature `0.0`. The exact-match VQA
accuracy is `semantic_accuracy == 100`; relaxed semantic accuracy is
`semantic_accuracy >= 80`. The judge script also reports `== 100` rates for
`completeness`, `factuality`, and `conciseness`.

Example:

```bash
python scripts/judge_vqa_predictions.py \
  --predictions-json outputs/final/slake_gemma4_clip_dinov3cls_top6/slake/rg_icl_dual_global_similarity/predictions.json \
  --output-jsonl outputs/judge/slake_main/slake_gemma4_clip_dinov3cls_top6_gpt54mini.jsonl \
  --summary-json outputs/judge/slake_main/slake_gemma4_clip_dinov3cls_top6_gpt54mini_summary.json \
  --prompt-path configs/judge/vqa_llm_judge_prompt_v1.txt \
  --model gpt-5.4-mini-2026-03-17 \
  --temperature 0 \
  --max-workers 100
```

After all judge summaries are present, generate the VQA result figures:

```bash
python scripts/plot_vqa_llm_judge_results.py
```

For VQA-Med2019, the same prompt and scoring rules are used through the Batch
API helper:

```bash
python scripts/run_vqamed2019_llm_judge_batch.py prepare
python scripts/run_vqamed2019_llm_judge_batch.py submit
python scripts/run_vqamed2019_llm_judge_batch.py status
python scripts/run_vqamed2019_llm_judge_batch.py download
python scripts/run_vqamed2019_llm_judge_batch.py convert
```

VQA-RAD uses the same Batch API wrapper and includes the Gemma4 LoRA projector
baseline in addition to the 18 main VQA runs:

```bash
python scripts/run_vqa_rad_llm_judge_batch.py prepare
python scripts/run_vqa_rad_llm_judge_batch.py submit
python scripts/run_vqa_rad_llm_judge_batch.py status
python scripts/run_vqa_rad_llm_judge_batch.py download
python scripts/run_vqa_rad_llm_judge_batch.py convert
```

For PathVQA and VQA-Med2019 LoRA judge batches, use:

```bash
python scripts/poll_lora_vqa_batch_judge.py --poll-interval 60 --max-poll-minutes 0
```

The VQA figure script reads LoRA summaries for SLAKE, PathVQA, VQA-Med2019,
and VQA-RAD when the corresponding `outputs/judge/*lora*summary.json` files
are present. The LoRA row is plotted in dark blue.

## Gemma4 VQA LoRA Baseline

The VQA LoRA baseline uses the same train-only supervised setup across
SLAKE, PathVQA, VQA-Med2019, and VQA-RAD:

```text
base model: google/gemma-4-31B-it
LoRA scope: language decoder + multimodal projector
r: 16
alpha: 32
dropout: 0.05
learning rate: 1e-4
epochs: 5
batch size: 1
gradient accumulation: 16
bf16 full-precision base model, no 4-bit quantization
eval/test max tokens: 512
checkpoint selection: validation balanced open-token-recall / closed accuracy
inference: merge LoRA into base model, serve merged model with vLLM
```

Run the end-to-end train, merge, vLLM inference, and cleanup pipeline with:

```bash
DATASET=pathvqa CUDA_DEVICES=0,1 PORT=18326 \
  bash scripts/run_gemma4_vqa_lora_projector512_pipeline.sh

DATASET=vqamed2019 CUDA_DEVICES=0,1 PORT=18366 \
  bash scripts/run_gemma4_vqa_lora_projector512_pipeline.sh

DATASET=vqa_rad CUDA_DEVICES=0,1 PORT=18376 \
  bash scripts/run_gemma4_vqa_lora_projector512_pipeline.sh
```

The same script accepts prompt overrides through `VQA_SYSTEM_PROMPT`,
`VQA_QUERY_TEMPLATE`, `JSON_INSTRUCTION`, and `ASSISTANT_FORMAT`, which are
used only when intentionally matching a published dataset-specific prompt.

## ResNet50 Training

BreaKHis 8-class multiclass:

```bash
bash scripts/run_breakhis_resnet50_multiclass_pooled.sh
bash scripts/run_breakhis_resnet50_multiclass_mag40_100.sh
bash scripts/run_breakhis_resnet50_multiclass_mag200_400.sh
```

DDR 512 image set with 384 input:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_resnet50_classification.py \
  --manifest-csv manifests/ddr_official_split_crop_pad_512.csv \
  --data-root . \
  --output-dir outputs/final/resnet50_ddr_crop_pad_512_384_seed3407 \
  --image-size 384 \
  --batch-size 32 \
  --epochs 40 \
  --patience 8 \
  --num-workers 8 \
  --seed 3407 \
  --backbone-lr 3e-5 \
  --head-lr 1e-3 \
  --weight-decay 1e-4 \
  --selection-metric accuracy
```

TBX11K supervised baselines:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_resnet50_classification.py \
  --manifest-csv manifests/tbx11k_train85_val15_officialval_test_seed3407.csv \
  --data-root . \
  --output-dir outputs/final/resnet50_tbx11k_384_seed3407 \
  --image-size 384 \
  --batch-size 32 \
  --epochs 40 \
  --patience 8 \
  --num-workers 8 \
  --seed 3407 \
  --backbone-lr 3e-5 \
  --head-lr 1e-3 \
  --weight-decay 1e-4 \
  --selection-metric accuracy

CUDA_VISIBLE_DEVICES=0 python scripts/train_vit_classification.py \
  --manifest-csv manifests/tbx11k_train85_val15_officialval_test_seed3407.csv \
  --data-root . \
  --output-dir outputs/final/vit224_tbx11k_seed3407 \
  --task single \
  --image-size 224 \
  --batch-size 128 \
  --epochs 200 \
  --patience 25 \
  --min-epochs 50 \
  --num-workers 8 \
  --seed 3407 \
  --backbone-lr 3e-5 \
  --head-lr 1e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --selection-metric accuracy
```

CheXpert5 multilabel:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_resnet50_chexpert_multilabel.py \
  --manifest-csv manifests/chexpert_official_split_320.csv \
  --data-root . \
  --output-dir outputs/final/resnet50_chexpert_multilabel_320_seed3407 \
  --image-size 320 \
  --batch-size 64 \
  --epochs 20 \
  --patience 5 \
  --num-workers 8 \
  --seed 3407 \
  --backbone-lr 3e-5 \
  --head-lr 1e-3 \
  --weight-decay 1e-4
```

## Result Files

Each VLM method writes:

```text
outputs/final/<run_name>/<dataset>/<method>/predictions.json
outputs/final/<run_name>/<dataset>/<method>/raw_outputs.jsonl
outputs/final/<run_name>/<dataset>/<method>/metrics.json
```

Each ResNet run writes checkpoints and final metrics under its `--output-dir`.

Each VQA judge run writes:

```text
outputs/judge/<dataset>_main/<run_name>_gpt54mini.jsonl
outputs/judge/<dataset>_main/<run_name>_gpt54mini_summary.json
outputs/figures/vqa_llm_judge_<dataset>_horizontal.pdf
outputs/figures/vqa_llm_judge_<dataset>.csv
```

Collect final metrics:

```bash
python scripts/collect_final_metrics.py --outputs-root outputs/final
```

## Reproducibility Checklist

Before reporting results, verify:

- all data paths in the manifest resolve on the local machine;
- BreaKHis patient IDs are split strictly at patient level;
- BreaKHis magnification-specific runs use the matching magnification manifest
  and fixed exemplar JSON;
- DDR VLM configs use the `crop_pad_512` manifest;
- CheXpert final configs are `chexpert5_*`, not legacy `chexpert_*` 14-label
  configs;
- current VQA-RAD main runs use `data/vqa/vqa_rad/manifest_refval.json`, fixed
  exemplars from `data/vqa/vqa_rad/fixed_exemplars_refval_seed3407.json`, and
  `outputs/features_*_vqa_rad_refval_promptfix` feature roots;
- VQA-RAD `_512` configs and `manifests/vqa_rad_official_split_512.csv` are
  retained only for the legacy exact-512 workflow;
- vLLM server ports match `inference.base_url`;
- `VLLM_API_KEY=EMPTY` is set for local vLLM;
- feature directories referenced in retrieval configs exist before launching
  retrieval-guided VLM runs;
- TBX11K final configs use `tbx11k_*`, the TBX11K train split is the only ICL reference pool, and the official TBX11K validation split is reported as test under this project protocol;
- current VQA configs use original image size with `image_max_side: null`;
- VQA LoRA inference uses `max_tokens: 512` and merged-LoRA vLLM serving;
- VQA judge uses `configs/judge/vqa_llm_judge_prompt_v1.txt` with SHA256
  `20d61c55dd8a5db11cce4fa3cfa5b13bcdf32eede2b3c447132c946375caba65`;
- VQA judge model is `gpt-5.4-mini-2026-03-17` with temperature `0.0`;
- `outputs/` and `logs/` are not committed to git.
