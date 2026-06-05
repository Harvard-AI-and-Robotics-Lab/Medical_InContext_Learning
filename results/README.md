# Results

This folder contains curated, paper-facing result tables and figures copied from
the working experiment directory. It does not contain raw VLM outputs, model
weights, logs, or embedding arrays.

Important distinction:

- `configs/final/` defines the fully aligned final protocol.
- `results/tables/current_experiment_summary_main_table.*` is a snapshot of the
  current working results. Some rows are historical and explicitly marked in
  the table notes if their temperature or probability protocol differed.

Before final submission, regenerate the table from runs produced by
`configs/final/` if every row needs to be strictly protocol-identical.

Main current best ICL result:

```text
Qwen3.6-27B, non-thinking, temperature=1.0,
k=6 references by 0.5 CLIP + 0.5 DINOv3 CLS:
Acc 0.9465, AUC 0.9752, CM [[451, 21], [18, 239]]
```
