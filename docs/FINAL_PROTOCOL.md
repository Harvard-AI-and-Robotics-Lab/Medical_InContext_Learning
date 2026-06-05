# Final Experimental Protocol

The formal main experiment matrix is defined in
`docs/FORMAL_EXPERIMENT_MATRIX.md`. Dataset-specific prompts, output schemas,
and probability semantics are defined in `docs/DATASET_CONTRACTS.md` and
`src/prompting/dataset_contracts.py`. This file records the current LAG-specific
implementation details for that matrix.

## Dataset

The LAG manifest in `manifests/lag_manifest.csv` contains:

| Split | non_glaucoma | glaucoma | Total |
| --- | ---: | ---: | ---: |
| train | 2200 | 1197 | 3397 |
| val | 471 | 257 | 728 |
| test | 472 | 257 | 729 |

The train split is the only reference pool for all ICL retrieval methods.
Validation and test images are never used as references.

## VLM Inference

Final VLM inference is run through vLLM with:

```yaml
temperature: 1.0
top_p: null
max_tokens: 512
response_format: json_object
enable_thinking: false
chat_template_kwargs:
  enable_thinking: false
seed: 3407
parallel_requests: 256
batch_size: 256
```

The final answer must be one compact JSON object:

```json
{
  "label": "glaucoma|non_glaucoma",
  "confidence": 0.0,
  "probability": 0.0,
  "evidence": "eight words max"
}
```

Definitions:

- `label`: final binary diagnosis.
- `confidence`: confidence in the chosen final label.
- `probability`: probability that the image is glaucoma.
- `evidence`: short visual evidence string; not used for metrics.

Classification metrics use `label`. Calibration and ranking metrics use
`probability`.

## ICL Reference Methods

All ICL methods use `k = 6`.

| Method | Reference selection | Similarity shown in prompt |
| --- | --- | --- |
| No-ICL | none | no |
| Fixed-6 | same six train images for every query | no |
| Random-6 | six train images sampled per query using `seed=3407 + hash(query_id)` | no |
| CLIP top-6 | top-6 train images by CLIP global cosine similarity | yes |
| DINOv3 top-6 | top-6 train images by DINOv3 CLS-token cosine similarity | yes |
| CLIP+DINOv3 top-6 | top-6 train images by equal-weight CLIP + DINOv3 CLS similarity | yes |

The fused retrieval score is:

```text
score(query, reference)
  = 0.5 * cosine(CLIP_query, CLIP_reference)
  + 0.5 * cosine(DINOv3_CLS_query, DINOv3_CLS_reference)
```

Implementation uses the equivalent cosine similarity of concatenated normalized
features:

```text
cosine(concat(sqrt(0.5)*L2(CLIP), sqrt(0.5)*L2(DINOv3_CLS)))
```

## Encoders

| Encoder | Model | Image size | Embedding |
| --- | --- | ---: | --- |
| CLIP | `openai/clip-vit-large-patch14` | 224 | 1024-d global |
| DINOv3 CLS | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 512 | 1024-d CLS |
| DINOv3 patch-mean | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 512 | exploratory ablation only |

## Fixed-6 References

The fixed six examples are from the train split:

| ID | Label |
| --- | --- |
| 0984 | non_glaucoma |
| 1759 | glaucoma |
| 2216 | glaucoma |
| 2438 | glaucoma |
| 2443 | glaucoma |
| 4665 | non_glaucoma |
