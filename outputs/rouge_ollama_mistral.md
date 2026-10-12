# 📊 Empirical ROUGE Benchmark Study Report

**Generated on:** 2026-10-08T19:24:28.527046Z  
**Evaluated Samples:** 15 documents from `data/eval/benchmark_with_summaries.json`  
**Backend:** `ollama`  
**Interpretation:** Interpret results together with backend and model details; scores depend on the selected model.

---

## 1. Comparative Results Table

| Method | Type | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-L F1 | ROUGE-L Recall |
|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Baseline | 0.4179 | 0.5133 | 0.1638 | 0.2047 | 0.2988 | 0.3684 |
| **YAKE Extractive** | Zero-LLM Proposed | 0.4090 | 0.5004 | 0.1569 | 0.1951 | 0.2910 | 0.3573 |
| **LLM Unconditioned** | Baseline Abstractive | 0.4342 | 0.6725 | 0.1649 | 0.2502 | 0.3260 | 0.5039 |
| **YAKE LLM Guided** | Full Pipeline | 0.4258 | 0.6507 | 0.1602 | 0.2443 | 0.3171 | 0.4855 |
| **Oracle LLM Guided** | Upper Bound (Gold KW) | 0.4454 | 0.6540 | 0.1842 | 0.2643 | 0.3545 | 0.5148 |

---

## 2. Uncertainty and length

With only 15 short documents, differences between methods are often within noise. Values below are
mean ROUGE-1/2/L F1 with a 95% bootstrap CI, the paired difference against Lead-3 (wins/ties/losses per
document) and the average summary length (reference: 36.7 words).

| Method | ROUGE-1 F1 [95% CI] | ROUGE-2 F1 [95% CI] | ROUGE-L F1 [95% CI] | Δ R1 vs Lead-3 (W/T/L) | Avg words |
|---|---|---|---|---|---|
| lead_3 | 0.418 [0.373, 0.473] | 0.164 [0.117, 0.217] | 0.299 [0.246, 0.367] | baseline | 53.7 |
| yake_extractive | 0.409 [0.363, 0.466] | 0.157 [0.111, 0.211] | 0.291 [0.238, 0.361] | -0.009 (1/12/2) | 53.5 |
| llm_unconditioned | 0.434 [0.409, 0.465] | 0.165 [0.138, 0.200] | 0.326 [0.303, 0.355] | +0.016 (8/0/7) | 79.5 |
| yake_llm_guided | 0.426 [0.390, 0.470] | 0.160 [0.125, 0.205] | 0.317 [0.269, 0.378] | +0.008 (8/0/7) | 78.1 |
| oracle_llm_guided | 0.445 [0.407, 0.492] | 0.184 [0.137, 0.241] | 0.354 [0.302, 0.421] | +0.028 (10/0/5) | 74.9 |

`*` = paired 95% CI excludes zero.

Reference provenance: 5 × `project_created`, 10 × `unverified`

---

## 3. Interpretation

- **Backend scope:** Interpret results together with backend and model details; scores depend on the selected model.
- **Extractive comparison:** Lead-3 and YAKE Extractive are directly comparable without an LLM service.
- **Abstractive comparison:** Run this study with a configured local model before drawing quality conclusions about keyword conditioning.
