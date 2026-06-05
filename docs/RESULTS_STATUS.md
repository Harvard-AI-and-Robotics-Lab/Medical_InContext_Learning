# Results Status

The final code path and configs are aligned. The copied result tables are a
current snapshot for writing and plotting, not a guarantee that every row was
generated under the final protocol.

Important BreaKHis note: results generated before the strict patient-code split
fix are stale. The old split grouped by full lesion/case suffix and could place
related cases such as `14-21998AB` and `14-21998EF` into different splits. Final
BreaKHis tables must be regenerated after `2026-05-09` using the strict patient
code manifest.

Already aligned with the final VLM protocol:

- Qwen3.6 CLIP+DINOv3 CLS top-6
- Gemma4 CLIP+DINOv3 CLS top-6
- Qwen3.6 Random-6
- Gemma4 Random-6
- Gemma4 CLIP top-6

Should be rerun with `configs/final/` before final paper tables if strict
protocol identity is required:

- Qwen3.6 No-ICL
- Gemma4 No-ICL
- Qwen3.6 Fixed-6
- Gemma4 Fixed-6
- Qwen3.6 CLIP top-6
- Qwen3.6 DINOv3 CLS top-6
- Gemma4 DINOv3 CLS top-6

The configs for these reruns are already present.
