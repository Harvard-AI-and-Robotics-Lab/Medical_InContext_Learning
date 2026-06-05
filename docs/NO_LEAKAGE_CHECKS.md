# Leakage Controls

The final LAG dataset wrapper uses:

```python
def get_reference_pool(self):
    return [s for s in self.samples if s.split == "train"]
```

All retrieval-based ICL methods build the retriever index only from
`dataset.get_reference_pool()`. The query set is `dataset.get_test_samples()`.

For global retrieval, the code filters feature rows to train sample IDs before
building the index:

```python
reference_pool = dataset.get_reference_pool()
ref_by_id = {sample.id: sample for sample in reference_pool}
reference_ids = set(ref_by_id.keys())
train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
filtered_embeddings = global_embeddings[train_indices]
```

Run:

```bash
python scripts/validate_final_setup.py
```

before final experiments to verify that split IDs do not overlap and the fixed
references are all train samples.
