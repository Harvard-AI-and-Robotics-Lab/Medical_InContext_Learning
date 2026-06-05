# Formal Experiment Matrix

This document is the fixed experiment contract for formal classification runs.
Exploratory studies, prompt variants, thinking-mode runs, k-sweeps, LoRA
experiments, and extra retrieval backbones must not be mixed into this table
unless this document is explicitly updated.

## Scope

For every classification dataset included in the formal paper experiments, run
the same experiment matrix below. If the dataset has an official split, use the
official train/validation/test split. If no official split is provided, use a
global patient-level random split with target ratio 0.70/0.15/0.15 and
`seed=3407`.

The train split is the only allowed reference pool for all ICL methods.
Validation and test samples must never be used as references.

## Generation Protocol

All VLM experiments use the aligned final protocol:

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

Output schemas are dataset-specific:

- Binary datasets use `label`, `confidence`, scalar `probability`, and
  `evidence`.
- Multi-class datasets use `label`, `confidence`, class-wise `probabilities`,
  and `evidence`.

The parser and metrics must interpret probabilities according to the dataset
schema. For example, LAG uses `probability = p_glaucoma`, while BreaKHis uses an
8-class `probabilities` distribution.

## Required Experiments

Each row below is required for each formal classification dataset.

| Category | Experiment | Models |
| --- | --- | --- |
| No-ICL | Query image only, no references | Qwen3.6, Gemma4 |
| Fixed-6 | Same six fixed train references for every query | Qwen3.6, Gemma4 |
| Random-6 | Six train references sampled per query | Qwen3.6, Gemma4 |
| CLIP top-6 | Top six train references by CLIP global similarity | Qwen3.6, Gemma4 |
| DINOv3 top-6 | Top six train references by DINOv3 CLS-token similarity | Qwen3.6, Gemma4 |
| CLIP+DINOv3 top-6 | Top six train references by equal-weight CLIP + DINOv3 similarity | Qwen3.6, Gemma4 |
| Supervised | ImageNet-pretrained ResNet50 fine-tuned on the train split | ResNet50 |

This gives 12 VLM runs plus 1 supervised baseline per dataset.

## Reference Method Definitions

### No-ICL

The model receives only the query image and the dataset-specific classification
prompt.

### Fixed-6

For each dataset, sample one fixed set of six train references using `seed=3407`.
The same six references are used for every query and for both VLMs. Fixed
references must be stored under `manifests/` and must be auditable.

Current LAG fixed references are stored in:

```text
manifests/fixed_exemplars_seed3407.json
```

For other datasets, use dataset-specific filenames, for example:

```text
manifests/breakhis_fixed_exemplars_seed3407.json
```

### Random-6

For each query, sample six references from the train split only. The sampling
seed is deterministic:

```text
query_seed = seed + hash(query_id)
```

The same query-specific references must be used for Qwen3.6 and Gemma4.

### CLIP top-6

Retrieve the top six train references by global cosine similarity from:

```text
openai/clip-vit-large-patch14
```

The prompt includes each reference label and its similarity score.

### DINOv3 top-6

Retrieve the top six train references by cosine similarity of DINOv3 CLS-token
embeddings:

```text
facebook/dinov3-vitl16-pretrain-lvd1689m
```

The CLS token is the global image embedding exported by the DINOv3 encoder and
then L2-normalized before cosine retrieval. The prompt includes each reference
label and its similarity score.

### CLIP+DINOv3 top-6

Retrieve the top six train references by equal-weight fused similarity:

```text
score(query, reference)
  = 0.5 * cosine(CLIP_query, CLIP_reference)
  + 0.5 * cosine(DINOv3_query, DINOv3_reference)
```

The current formal fused setting uses CLIP global embeddings plus DINOv3 CLS
embeddings. If this is changed to DINOv3 patch-mean fusion, update this document
and all configs before running.

The prompt includes each reference label and its fused similarity score.

### Supervised ResNet50

Train an ImageNet-pretrained ResNet50 on the train split only. Select the best
checkpoint using validation performance, then evaluate once on the held-out test
split.

## Formal vs Exploratory

The following are exploratory unless explicitly promoted into this document:

- thinking-mode VLM runs;
- `temperature` or `top_p` sweeps;
- `k` sweeps such as k=4/8/10;
- extra retrieval encoders such as MAE, SigLIP, OpenCLIP, or BiomedCLIP;
- LoRA/SFT experiments;
- kNN-only retrieval baselines;
- prompt variants that change output schema or evidence instructions.

Exploratory results can be reported separately, but they are not part of the
formal main experiment matrix.
