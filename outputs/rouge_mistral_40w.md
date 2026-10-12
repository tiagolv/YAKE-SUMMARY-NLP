# 📊 Empirical ROUGE Benchmark Study Report

**Generated on:** 2026-10-08T20:25:20.757459Z  
**Evaluated Samples:** 15 documents from `data/eval/benchmark_with_summaries.json`  
**Backend:** `ollama` (model `mistral`)  
**LLM length limit:** 40 words (prompted)  
**Interpretation:** Interpret results together with backend and model details; scores depend on the selected model.

---

## 1. Comparative Results Table

| Method | Type | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-L F1 | ROUGE-L Recall |
|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Baseline | 0.4179 | 0.5133 | 0.1638 | 0.2047 | 0.2988 | 0.3684 |
| **YAKE Extractive** | Zero-LLM Proposed | 0.4090 | 0.5004 | 0.1569 | 0.1951 | 0.2910 | 0.3573 |
| **LLM Unconditioned** | Baseline Abstractive | 0.4715 | 0.6107 | 0.1869 | 0.2411 | 0.3571 | 0.4620 |
| **YAKE LLM Guided** | Full Pipeline | 0.4686 | 0.6210 | 0.1888 | 0.2467 | 0.3565 | 0.4711 |
| **Oracle LLM Guided** | Upper Bound (Gold KW) | 0.4691 | 0.6138 | 0.1840 | 0.2383 | 0.3628 | 0.4730 |

---

## 2. Uncertainty and length

With only 15 short documents, differences between methods are often within noise. Values below are
mean ROUGE-1/2/L F1 with a 95% bootstrap CI, the paired difference against Lead-3 (wins/ties/losses per
document) and the average summary length (reference: 36.7 words).

| Method | ROUGE-1 F1 [95% CI] | ROUGE-2 F1 [95% CI] | ROUGE-L F1 [95% CI] | Δ R1 vs Lead-3 (W/T/L) | Avg words |
|---|---|---|---|---|---|
| lead_3 | 0.418 [0.373, 0.473] | 0.164 [0.117, 0.217] | 0.299 [0.246, 0.367] | baseline | 53.7 |
| yake_extractive | 0.409 [0.363, 0.466] | 0.157 [0.111, 0.211] | 0.291 [0.238, 0.361] | -0.009 (1/12/2) | 53.5 |
| llm_unconditioned | 0.471 [0.434, 0.513] | 0.187 [0.151, 0.230] | 0.357 [0.323, 0.398] | +0.053 * (11/0/4) | 59.8 |
| yake_llm_guided | 0.469 [0.431, 0.516] | 0.189 [0.145, 0.243] | 0.356 [0.319, 0.404] | +0.051 * (12/0/3) | 61.8 |
| oracle_llm_guided | 0.469 [0.435, 0.509] | 0.184 [0.147, 0.228] | 0.363 [0.319, 0.419] | +0.051 * (11/0/4) | 61.2 |

`*` = paired 95% CI excludes zero.

Reference provenance: 5 × `project_created`, 10 × `unverified`

---

## 3. Interpretation

- **Backend scope:** Interpret results together with backend and model details; scores depend on the selected model.
- **Extractive comparison:** Lead-3 and YAKE Extractive are directly comparable without an LLM service.
- **Abstractive comparison:** Run this study with a configured local model before drawing quality conclusions about keyword conditioning.
