# Plot Method Rows

This note makes the paper-figure method rows explicit. It addresses the
apparent mismatch between the compact runnable config list and the rows shown in
the plots.

The compact VLM suite uses eight config files for Qwen3.6 and Gemma4:

- Qwen3.6 no-ICL/fixed/random config
- Qwen3.6 CLIP top-6 config
- Qwen3.6 DINOv3 top-6 config
- Qwen3.6 CLIP+DINOv3 top-6 config
- Gemma4 no-ICL/fixed/random config
- Gemma4 CLIP top-6 config
- Gemma4 DINOv3 top-6 config
- Gemma4 CLIP+DINOv3 top-6 config

The two no-ICL/fixed/random configs each emit three plotted method rows:
`zero_shot`, `fixed_random_6`, and `random_icl`. Therefore the eight VLM config
files expand to twelve VLM plot rows.

When the Gemma4 language-LoRA baseline is included, the plot has thirteen
non-supervised rows. ViT and ResNet50 are separate supervised baselines.

| Plot row | Reproducible command source |
| --- | --- |
| Qwen3.6 No-ICL | `run_final_classification.py --config <qwen>_noicl_fixed6_random6.yaml --methods zero_shot` |
| Qwen3.6 Fixed-6 | `run_final_classification.py --config <qwen>_noicl_fixed6_random6.yaml --methods fixed_random_6` |
| Qwen3.6 Random-6 | `run_final_classification.py --config <qwen>_noicl_fixed6_random6.yaml --methods random_icl` |
| Qwen3.6 CLIP top-6 | `run_final_classification.py --config <qwen>_clip_top6.yaml` |
| Qwen3.6 DINOv3 top-6 | `run_final_classification.py --config <qwen>_dinov3_cls_top6.yaml` |
| Qwen3.6 CLIP+DINOv3 top-6 | `run_final_classification.py --config <qwen>_clip_dinov3cls_top6.yaml` |
| Gemma4 No-ICL | `run_final_classification.py --config <gemma>_noicl_fixed6_random6.yaml --methods zero_shot` |
| Gemma4 Fixed-6 | `run_final_classification.py --config <gemma>_noicl_fixed6_random6.yaml --methods fixed_random_6` |
| Gemma4 Random-6 | `run_final_classification.py --config <gemma>_noicl_fixed6_random6.yaml --methods random_icl` |
| Gemma4 CLIP top-6 | `run_final_classification.py --config <gemma>_clip_top6.yaml` |
| Gemma4 DINOv3 top-6 | `run_final_classification.py --config <gemma>_dinov3_cls_top6.yaml` |
| Gemma4 CLIP+DINOv3 top-6 | `run_final_classification.py --config <gemma>_clip_dinov3cls_top6.yaml` |
| Gemma4 language LoRA SFT | `train_gemma4_language_lora.py` plus `eval_gemma4_language_lora.py` |

The helper below expands these rows explicitly:

```bash
scripts/run_plot_method_suite.sh lag all
scripts/run_plot_method_suite.sh tbx11k all
scripts/run_plot_method_suite.sh ddr_512 all
scripts/run_plot_method_suite.sh breakhis_binary all
```

Use the second argument to run a subset:

```bash
scripts/run_plot_method_suite.sh tbx11k vlm
scripts/run_plot_method_suite.sh tbx11k qwen
scripts/run_plot_method_suite.sh tbx11k gemma
scripts/run_plot_method_suite.sh tbx11k lora
```

For LoRA, the default generation test uses `max_new_tokens=512`, matching the
main VLM `max_tokens` setting. A log-probability evaluation can also be run for
ranking metrics:

```bash
LORA_LOGPROB_AUC=1 scripts/run_plot_method_suite.sh tbx11k lora
```

The LAG command uses the paper-facing dataset name `lag`, but the LoRA trainer
uses the internal preset key `lag_project`; the helper maps this automatically.
Run this smoke test after edits to verify the row expansion and LoRA defaults:

```bash
python scripts/validate_plot_method_suite.py
```
