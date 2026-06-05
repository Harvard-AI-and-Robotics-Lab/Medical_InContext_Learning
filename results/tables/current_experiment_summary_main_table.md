# LAG Glaucoma Experiment Summary

This table was generated from current working-directory metric snapshots. `CM` is `[[TN, FP], [FN, TP]]` for binary runs.

| Category | Model | Setting | Acc | AUC | Brier | ECE | CM | N | Comparable? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised baseline | ResNet50 | Fine-tuned ImageNet ResNet50, LAG train, 512px, threshold 0.5 | 0.9767 | 0.9985 | 0.0172 |  | [[457,15],[2,255]] | 729 | SFT baseline | Best supervised CNN; not ICL |
| Supervised baseline | ResNet50 | Fine-tuned ImageNet ResNet50, Youden threshold | 0.9753 | 0.9985 | 0.0172 |  | [[456,16],[2,255]] | 729 | SFT baseline | Same model, validation-derived threshold |
| ICL current best | Qwen3.6 27B | Non-thinking, temp=1, JSON prob, top-6 by 0.5 CLIP + 0.5 DINOv3 CLS, b=256 | 0.9465 | 0.9752 | 0.0475 | 0.0329 | [[451,21],[18,239]] | 729 | YES - current main ICL | Current best VLM-ICL |
| ICL current best | Gemma4 31B | Non-thinking, temp=1, JSON prob, top-6 by 0.5 CLIP + 0.5 DINOv3 CLS, b=256 | 0.9328 | 0.9662 | 0.0581 | 0.0511 | [[455,17],[32,225]] | 729 | YES | Same retrieval as current best |
| ICL CLIP-only | Qwen3.6 27B | Non-thinking, temp=0.7/top_p=0.8, JSON prob + similarity, CLIP global top-6 | 0.9273 | 0.9736 | 0.0598 | 0.0612 | [[442,30],[23,234]] | 729 | Partly - older temp | Strong CLIP-only baseline; probability/similarity included |
| ICL CLIP-only | Gemma4 31B | Non-thinking, temp=1, JSON prob + similarity, CLIP global top-6 | 0.9246 | 0.9712 | 0.0628 | 0.0623 | [[448,24],[31,226]] | 729 | YES-ish | Comparable temp=1; CLIP-only |
| ICL CLIP-only | Gemma4 31B | Non-thinking, temp=0, JSON prob + similarity, CLIP global top-6 | 0.9232 | 0.9754 | 0.0631 | 0.0653 | [[447,25],[31,226]] | 729 | Partly - temp differs | Earlier probability figure run |
| Thinking | Qwen3.6 27B | Thinking on, budget=2048, temp=1, JSON prob + similarity, CLIP global top-6, b=256 | 0.9163 | 0.9646 | 0.0691 | 0.0479 | [[438,34],[27,230]] | 729 | YES for thinking ablation | Thinking hurt vs non-thinking CLIP-only/current best |
| Thinking | Gemma4 31B | Thinking on, budget=2048, temp=1, JSON prob + similarity, CLIP global top-6, b=256 | 0.9259 | 0.9646 | 0.0630 | 0.0548 | [[442,30],[24,233]] | 729 | YES for thinking ablation | Slightly above Gemma non-thinking acc, lower AUC |
| Zero-shot | Qwen3.6 27B | Non-thinking, temp=0.7/top_p=0.8, JSON probability | 0.7805 | 0.8510 | 0.1795 | 0.1280 | [[342,130],[30,227]] | 729 | Partly - older temp | No references |
| Zero-shot | Gemma4 31B | Non-thinking, temp=0, JSON probability | 0.7572 | 0.8079 | 0.1993 | 0.1589 | [[377,95],[82,175]] | 729 | Partly - temp differs | No references; earlier probability figure run |
| Fixed random | Gemma4 31B | Non-thinking, temp=0, JSON probability, fixed random-6 | 0.7874 | 0.8278 | 0.1724 | 0.1058 | [[372,100],[55,202]] | 729 | Partly - temp differs | Earlier probability figure run |
| KNN correction | Qwen3.6 27B | Non-thinking, temp=0.7, CLIP top-6 + similarity + kNN prediction, model audits override | 0.9191 | 0.9527 | 0.0669 | 0.0539 | [[424,48],[11,246]] | 729 | Exploratory | Did not beat normal CLIP top-6 |
| Balanced 3+3 | Qwen3.6 27B | Non-thinking, temp=0.7, top-3 glaucoma + top-3 non-glaucoma CLIP refs + similarity | 0.8615 | 0.9059 | 0.1107 | 0.0762 | [[393,79],[22,235]] | 729 | Exploratory | Lower acc; more FP |
| Encoder ablation | Qwen3.6 27B | OpenCLIP ViT-bigG/14 global top-6, temp=0.7, JSON prob + similarity | 0.9273 | 0.9691 | 0.0603 | 0.0457 | [[431,41],[12,245]] | 729 | Exploratory | Tied CLIP-only acc, lower AUC |
| Encoder ablation | Qwen3.6 27B | OpenCLIP ViT-H/14 global top-6, temp=0.7, JSON prob + similarity | 0.9204 | 0.9582 | 0.0662 | 0.0434 | [[433,39],[19,238]] | 729 | Exploratory | Below CLIP-only |
| Encoder ablation | Qwen3.6 27B | SigLIP2 SO400M global top-6, temp=0.7, JSON prob + similarity | 0.9177 | 0.9574 | 0.0687 | 0.0419 | [[433,39],[21,236]] | 729 | Exploratory | Below CLIP-only |
| Encoder ablation | Qwen3.6 27B | BiomedCLIP global top-6, temp=0.7, JSON prob + similarity | 0.9108 | 0.9555 | 0.0747 | 0.0402 | [[437,35],[30,227]] | 729 | Exploratory | Below CLIP-only |
| Deprecated old protocol | Qwen3.6 27B | Old rg_icl_global CLIP top-6, no explicit similarity/probability protocol | 0.9191 | 0.3649 | 0.5673 | 0.5752 | [[438,34],[25,232]] | 729 | NO | Accuracy usable-ish; AUC invalid/not comparable |
| Deprecated old protocol | Qwen3.6 27B | Old zero-shot in combined run, no explicit probability protocol | 0.8080 | 0.5606 | 0.4830 | 0.5090 | [[394,78],[62,195]] | 729 | NO | AUC/probability not comparable |
| Deprecated old protocol | Qwen3.6 27B | Old fixed_random_6 in combined run | 0.6365 | 0.5039 | 0.5831 | 0.5960 | [[461,11],[254,3]] | 729 | NO | Poor/format issue; not main |
| Deprecated local Gemma | Gemma4 local | Old local zero-shot, fp/bf variants | 0.7819 | 0.7634 | 0.1743 | 0.0991 | [[409,63],[96,161]] | 729 | NO | Original ~78% baseline; old protocol |
| Deprecated local Gemma | Gemma4 local | Old local CLIP global top-6 | 0.9246 | 0.9576 | 0.0650 | 0.0492 | [[447,25],[30,227]] | 729 | NO | Useful historical run, but no explicit JSON probability/similarity protocol |
