"""Automated ROUGE evaluation benchmark study.

Compares:
1. Lead-3 (standard extractive baseline)
2. YAKE Extractive (our lightweight, zero-LLM method)
3. LLM Unconditioned (abstractive without keywords)
4. YAKE LLM Guided (our full guided abstractive method)
5. Oracle LLM Guided (guided by human gold keywords)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yake_sum.abstractive.summarizer import AbstractiveSummarizer
from yake_sum.backends import BaseLLMClient, get_llm_client
from yake_sum.evaluation.rouge import compute_rouge_metrics
from yake_sum.extractive.summarizer import ExtractiveSummarizer, split_into_sentences


def compute_lead_3(text: str) -> str:
    """Extract first 3 sentences as standard baseline."""
    sentences = split_into_sentences(text)
    return " ".join(sentences[:3])


def run_rouge_benchmark(
    benchmark_file: Path,
    backend: str = "mock",
    model: str = "mistral",
    max_samples: int | None = None,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, Any]:
    data = json.loads(benchmark_file.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    if max_samples:
        documents = documents[:max_samples]

    client = llm_client or get_llm_client(backend=backend, model=model)
    extractive = ExtractiveSummarizer(num_sentences=3)
    abstractive = AbstractiveSummarizer(llm_client=client)

    methods = [
        "lead_3",
        "yake_extractive",
        "llm_unconditioned",
        "yake_llm_guided",
        "oracle_llm_guided",
    ]

    method_scores: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    sample_ids: list[str] = []

    for doc in documents:
        doc_id = doc["id"]
        doc_text = doc["document"]
        ref_summary = doc["reference_summary"]
        gold_keywords = doc.get("gold_keywords", [])
        sample_ids.append(doc_id)

        # 1. Lead-3
        lead_summary = compute_lead_3(doc_text)
        lead_rouge = compute_rouge_metrics(summary=lead_summary, reference=ref_summary)
        method_scores["lead_3"].append(lead_rouge)

        # 2. YAKE Extractive
        yake_ext_res = extractive.summarize(doc_text)
        yake_ext_rouge = compute_rouge_metrics(
            summary=yake_ext_res.summary, reference=ref_summary
        )
        method_scores["yake_extractive"].append(yake_ext_rouge)

        # 3. LLM Unconditioned
        llm_uncond_res = abstractive.summarize(doc_text, use_keywords=False)
        llm_uncond_rouge = compute_rouge_metrics(
            summary=llm_uncond_res.summary, reference=ref_summary
        )
        method_scores["llm_unconditioned"].append(llm_uncond_rouge)

        # 4. YAKE LLM Guided
        yake_abs_res = abstractive.summarize(doc_text, use_keywords=True)
        yake_abs_rouge = compute_rouge_metrics(
            summary=yake_abs_res.summary, reference=ref_summary
        )
        method_scores["yake_llm_guided"].append(yake_abs_rouge)

        # 5. Oracle LLM Guided
        oracle_res = abstractive.summarize(
            doc_text, external_keywords=gold_keywords, use_keywords=True
        )
        oracle_rouge = compute_rouge_metrics(
            summary=oracle_res.summary, reference=ref_summary
        )
        method_scores["oracle_llm_guided"].append(oracle_rouge)

    # Compute averages across all samples
    averaged_methods: dict[str, dict[str, float]] = {}
    for m in methods:
        scores_list = method_scores[m]
        n = len(scores_list) if scores_list else 1
        averaged_methods[m] = {
            "rouge1_p": round(sum(s["rouge1_p"] for s in scores_list) / n, 4),
            "rouge1_r": round(sum(s["rouge1_r"] for s in scores_list) / n, 4),
            "rouge1_f": round(sum(s["rouge1_f"] for s in scores_list) / n, 4),
            "rouge2_p": round(sum(s["rouge2_p"] for s in scores_list) / n, 4),
            "rouge2_r": round(sum(s["rouge2_r"] for s in scores_list) / n, 4),
            "rouge2_f": round(sum(s["rouge2_f"] for s in scores_list) / n, 4),
            "rougeL_p": round(sum(s["rougeL_p"] for s in scores_list) / n, 4),
            "rougeL_r": round(sum(s["rougeL_r"] for s in scores_list) / n, 4),
            "rougeL_f": round(sum(s["rougeL_f"] for s in scores_list) / n, 4),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backend": backend,
        "sample_ids": sample_ids,
        "methods": averaged_methods,
        "raw_scores": method_scores,
    }


def format_markdown_report(results: dict[str, Any]) -> str:
    m = results["methods"]
    n_docs = len(results["sample_ids"])
    backend = results["backend"]
    ts = results["timestamp"]
    interpretation = (
        "This is a deterministic pipeline smoke test. Mock outputs must not be used "
        "as evidence of abstractive summary quality or keyword-conditioning gains."
        if backend == "mock"
        else "Interpret results together with backend and model details; scores depend on the selected model."
    )

    report = f"""# 📊 Empirical ROUGE Benchmark Study Report

**Generated on:** {ts}  
**Evaluated Samples:** {n_docs} documents from `data/eval/benchmark_with_summaries.json`  
**Backend:** `{backend}`  
**Interpretation:** {interpretation}

---

## 1. Comparative Results Table

| Method | Type | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-L F1 | ROUGE-L Recall |
|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Baseline | {m['lead_3']['rouge1_f']:.4f} | {m['lead_3']['rouge1_r']:.4f} | {m['lead_3']['rouge2_f']:.4f} | {m['lead_3']['rouge2_r']:.4f} | {m['lead_3']['rougeL_f']:.4f} | {m['lead_3']['rougeL_r']:.4f} |
| **YAKE Extractive** | Zero-LLM Proposed | {m['yake_extractive']['rouge1_f']:.4f} | {m['yake_extractive']['rouge1_r']:.4f} | {m['yake_extractive']['rouge2_f']:.4f} | {m['yake_extractive']['rouge2_r']:.4f} | {m['yake_extractive']['rougeL_f']:.4f} | {m['yake_extractive']['rougeL_r']:.4f} |
| **LLM Unconditioned** | Baseline Abstractive | {m['llm_unconditioned']['rouge1_f']:.4f} | {m['llm_unconditioned']['rouge1_r']:.4f} | {m['llm_unconditioned']['rouge2_f']:.4f} | {m['llm_unconditioned']['rouge2_r']:.4f} | {m['llm_unconditioned']['rougeL_f']:.4f} | {m['llm_unconditioned']['rougeL_r']:.4f} |
| **YAKE LLM Guided** | Full Pipeline | **{m['yake_llm_guided']['rouge1_f']:.4f}** | **{m['yake_llm_guided']['rouge1_r']:.4f}** | **{m['yake_llm_guided']['rouge2_f']:.4f}** | **{m['yake_llm_guided']['rouge2_r']:.4f}** | **{m['yake_llm_guided']['rougeL_f']:.4f}** | **{m['yake_llm_guided']['rougeL_r']:.4f}** |
| **Oracle LLM Guided** | Upper Bound (Gold KW) | {m['oracle_llm_guided']['rouge1_f']:.4f} | {m['oracle_llm_guided']['rouge1_r']:.4f} | {m['oracle_llm_guided']['rouge2_f']:.4f} | {m['oracle_llm_guided']['rouge2_r']:.4f} | {m['oracle_llm_guided']['rougeL_f']:.4f} | {m['oracle_llm_guided']['rougeL_r']:.4f} |

---

## 2. Interpretation

- **Backend scope:** {interpretation}
- **Extractive comparison:** Lead-3 and YAKE Extractive are directly comparable without an LLM service.
- **Abstractive comparison:** Run this study with a configured local model before drawing quality conclusions about keyword conditioning.
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROUGE benchmark study.")
    parser.add_argument(
        "--dataset",
        default="data/eval/benchmark_with_summaries.json",
        help="Path to benchmark JSON.",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "ollama", "openai_compatible"],
        help="Backend to run abstractive methods with.",
    )
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/rouge_benchmark_report.md",
        help="Path to write report.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.dataset)
    results = run_rouge_benchmark(
        benchmark_file=benchmark_path,
        backend=args.backend,
        model=args.model,
        max_samples=args.max_samples,
    )

    report_md = format_markdown_report(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    # Also save raw JSON
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Benchmark completed successfully! Report saved to {out_path}")
    print(f"Raw results saved to {json_path}")
    print("\n" + report_md)


if __name__ == "__main__":
    main()
