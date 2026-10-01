# 📊 Empirical ROUGE Benchmark Study Report

**Generated on:** 2026-10-01T13:52:53.264862Z  
**Evaluated Samples:** 15 documents from `data/eval/benchmark_with_summaries.json`  
**Backend:** `mock`  
**Interpretation:** This is a deterministic pipeline smoke test. Mock outputs must not be used as evidence of abstractive summary quality or keyword-conditioning gains.

---

## 1. Comparative Results Table

| Method | Type | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-L F1 | ROUGE-L Recall |
|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Baseline | 0.4179 | 0.5133 | 0.1638 | 0.2047 | 0.2988 | 0.3684 |
| **YAKE Extractive** | Zero-LLM Proposed | 0.4090 | 0.5004 | 0.1569 | 0.1951 | 0.2910 | 0.3573 |
| **LLM Unconditioned** | Baseline Abstractive | 0.2462 | 0.1883 | 0.0857 | 0.0657 | 0.1898 | 0.1456 |
| **YAKE LLM Guided** | Full Pipeline | 0.2160 | 0.2083 | 0.0689 | 0.0672 | 0.1702 | 0.1645 |
| **Oracle LLM Guided** | Upper Bound (Gold KW) | 0.3349 | 0.3114 | 0.1255 | 0.1178 | 0.2270 | 0.2110 |

---

## 2. Uncertainty and length

With only 15 short documents, differences between methods are often within noise. Values below are
mean ROUGE-1/2/L F1 with a 95% bootstrap CI, the paired difference against Lead-3 (wins/ties/losses per
document) and the average summary length (reference: 36.7 words).

| Method | ROUGE-1 F1 [95% CI] | ROUGE-2 F1 [95% CI] | ROUGE-L F1 [95% CI] | Δ R1 vs Lead-3 (W/T/L) | Avg words |
|---|---|---|---|---|---|
| lead_3 | 0.418 [0.373, 0.473] | 0.164 [0.117, 0.217] | 0.299 [0.246, 0.367] | baseline | 53.7 |
| yake_extractive | 0.409 [0.363, 0.466] | 0.157 [0.111, 0.211] | 0.291 [0.238, 0.361] | -0.009 (1/12/2) | 53.5 |
| llm_unconditioned | 0.246 [0.189, 0.314] | 0.086 [0.050, 0.130] | 0.190 [0.142, 0.246] | -0.172 * (1/0/14) | 19.3 |
| yake_llm_guided | 0.216 [0.173, 0.266] | 0.069 [0.041, 0.103] | 0.170 [0.132, 0.213] | -0.202 * (0/0/15) | 34.3 |
| oracle_llm_guided | 0.335 [0.298, 0.375] | 0.126 [0.099, 0.154] | 0.227 [0.185, 0.272] | -0.083 * (3/0/12) | 31.3 |

`*` = paired 95% CI excludes zero.

Reference provenance: 5 × `project_created`, 10 × `unverified`

---

## 3. Interpretation

- **Backend scope:** This is a deterministic pipeline smoke test. Mock outputs must not be used as evidence of abstractive summary quality or keyword-conditioning gains.
- **Extractive comparison:** Lead-3 and YAKE Extractive are directly comparable without an LLM service.
- **Abstractive comparison:** Run this study with a configured local model before drawing quality conclusions about keyword conditioning.
