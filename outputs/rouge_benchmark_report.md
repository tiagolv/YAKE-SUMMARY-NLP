# 📊 Empirical ROUGE Benchmark Study Report

**Generated on:** 2026-09-28T15:49:49.703806Z  
**Evaluated Samples:** 10 documents from `data/eval/benchmark_with_summaries.json`  
**Backend:** `mock`  
**Interpretation:** This is a deterministic pipeline smoke test. Mock outputs must not be used as evidence of abstractive summary quality or keyword-conditioning gains.

---

## 1. Comparative Results Table

| Method | Type | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-L F1 | ROUGE-L Recall |
|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Baseline | 0.4116 | 0.4865 | 0.1400 | 0.1668 | 0.2868 | 0.3394 |
| **YAKE Extractive** | Zero-LLM Proposed | 0.4009 | 0.4674 | 0.1309 | 0.1551 | 0.2962 | 0.3461 |
| **LLM Unconditioned** | Baseline Abstractive | 0.2964 | 0.2251 | 0.1152 | 0.0881 | 0.2297 | 0.1749 |
| **YAKE LLM Guided** | Full Pipeline | **0.2549** | **0.2445** | **0.0926** | **0.0904** | **0.2057** | **0.1978** |
| **Oracle LLM Guided** | Upper Bound (Gold KW) | 0.3556 | 0.3262 | 0.1357 | 0.1261 | 0.2668 | 0.2459 |

---

## 2. Interpretation

- **Backend scope:** This is a deterministic pipeline smoke test. Mock outputs must not be used as evidence of abstractive summary quality or keyword-conditioning gains.
- **Extractive comparison:** Lead-3 and YAKE Extractive are directly comparable without an LLM service.
- **Abstractive comparison:** Run this study with a configured local model before drawing quality conclusions about keyword conditioning.
