# Dataset Contracts

The formal pipeline keeps dataset-specific task logic out of generic prompting and
inference code.

Each classification dataset must define a contract in
`src/prompting/dataset_contracts.py`. The contract owns:

- system prompt
- No-ICL / ICL / retrieval-ICL method instructions
- reference-image label wording
- output JSON schema
- probability semantics
- retrieval prompt wording

The generic runners only call the active dataset contract through
`ClassificationTemplate`, `GlobalSimilarityPrompter`, `DualGlobalSimilarityPrompter`,
and `OutputParser`.

## Current Contracts

### LAG

- Task: binary glaucoma classification.
- Labels: `non_glaucoma`, `glaucoma`.
- Probability field: scalar `probability`, defined as `P(glaucoma)`.
- Reference label wording: `Diagnosis`.

### BreaKHis

- Task: eight-class breast histopathology tumor-subtype classification.
- Labels: `adenosis`, `fibroadenoma`, `phyllodes_tumor`, `tubular_adenoma`,
  `ductal_carcinoma`, `lobular_carcinoma`, `mucinous_carcinoma`,
  `papillary_carcinoma`.
- Probability field: `probabilities`, a calibrated distribution over all eight
  subtype labels.
- Reference label wording: `Tumor subtype`.

## Adding A Dataset

Add a new `ClassificationDatasetContract` entry instead of adding `if dataset`
branches to generic prompt builders. If a dataset has a different output format,
add the interpretation to the contract and parser interface, not to individual
experiment scripts.
