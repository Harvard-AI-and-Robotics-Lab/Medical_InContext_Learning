# SLAKE VQA Protocol

This dataset is handled as a VQA task, separate from the classification dataset folder.

## Data

```bash
python3 scripts/prepare_slake_dataset.py --lang en --output-root data/vqa/slake
python3 scripts/validate_slake_vqa_setup.py
```

The English manifest is `data/vqa/slake/manifest_en.json`.
The reference pool is the SLAKE train split, validation remains validation, and test remains test.
Images are passed as original image bytes; SLAKE VQA configs set `image_max_side: null`.

## Features

Run this before retrieval methods:

```bash
scripts/run_slake_feature_pipeline.sh
```

This creates:

- `outputs/features_clip_global_slake/slake/clip`
- `outputs/features_dinov3_global_slake/slake/dinov3`
- `outputs/features_clip_dinov3cls_05_global_slake/slake/clip_dinov3cls05`

## vLLM Servers

The planned three-server layout uses two GPUs per server:

```bash
scripts/start_slake_vllm_servers.sh
```

Defaults:

- Qwen3.6: GPUs `2,3`, port `18101`, served name `qwen36-27b-slake`
- Gemma4: GPUs `4,5`, port `18102`, served name `gemma4-31b-slake`
- MedGemma: GPUs `6,7`, port `18103`, served name `medgemma-27b-slake`
- Local model directories under `/data/home/mindazhao/hf_models` are preferred when present.
- `MAX_MODEL_LEN=8192`, `MAX_NUM_BATCHED_TOKENS=8192`, and `--generation-config vllm` are used by default.

Check readiness:

```bash
curl -s -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" http://127.0.0.1:18101/v1/models
curl -s -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" http://127.0.0.1:18102/v1/models
curl -s -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" http://127.0.0.1:18103/v1/models
```

## Evaluation

Smoke test first:

```bash
LIMIT=3 scripts/run_slake_vqa_suite.sh qwen36
```

Full run:

```bash
scripts/run_slake_vqa_suite.sh all
```

Outputs are written under `outputs/final/slake_*`. Each method directory saves:

- `predictions.json`
- `raw_outputs.jsonl`
- `metrics.json` through the parent dataset metrics file

## LLM Judge

After raw responses are complete:

```bash
python3 scripts/judge_vqa_predictions.py \
  --predictions-json outputs/final/slake_qwen36_noicl_fixed6_random6/slake/zero_shot/predictions.json \
  --model gpt-5 \
  --max-workers 8
```

Use `--include-image` for an image-aware judge. By default this keeps original image bytes unless
`--image-max-side` is set.
