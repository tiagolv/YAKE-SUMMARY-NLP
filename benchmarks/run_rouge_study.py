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
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yake_sum.abstractive.summarizer import AbstractiveSummarizer
from yake_sum.backends import BaseLLMClient, get_llm_client
from yake_sum.evaluation.rouge import compute_rouge_metrics
from yake_sum.extractive.summarizer import ExtractiveSummarizer, split_into_sentences


VALID_REFERENCE_SOURCES = {"author_abstract", "curated_reference", "project_created", "unverified"}
MIN_DOCUMENTS = 15
_METRICS = ("rouge1_f", "rouge2_f", "rougeL_f")


def validate_dataset(data: dict[str, Any], min_documents: int = MIN_DOCUMENTS) -> list[dict[str, Any]]:
    """Validate the benchmark schema and return the documents; raise ValueError on problems."""
    docs = data.get("documents")
    if not isinstance(docs, list):
        raise ValueError("Dataset must contain a 'documents' list.")
    if len(docs) < min_documents:
        raise ValueError(f"Dataset has {len(docs)} documents; at least {min_documents} required.")
    seen: set[str] = set()
    for i, doc in enumerate(docs):
        for key in ("id", "document", "reference_summary", "reference_source"):
            if not isinstance(doc.get(key), str) or not doc[key].strip():
                raise ValueError(f"Document #{i}: field '{key}' is missing or empty.")
        if doc["id"] in seen:
            raise ValueError(f"Duplicate document id '{doc['id']}'.")
        seen.add(doc["id"])
        if doc["reference_source"] not in VALID_REFERENCE_SOURCES:
            raise ValueError(
                f"Document '{doc['id']}': invalid reference_source '{doc['reference_source']}'."
            )
        if not isinstance(doc.get("gold_keywords", []), list):
            raise ValueError(f"Document '{doc['id']}': gold_keywords must be a list.")
    return docs


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI of the mean (returns (0, 0) for empty input)."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return round(lo, 4), round(hi, 4)


def paired_comparison(a: list[float], b: list[float]) -> dict[str, Any]:
    """Per-document paired comparison of ``a`` vs ``b`` (mean diff, CI, wins/ties/losses)."""
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs)
    return {
        "mean_diff": round(sum(diffs) / len(diffs), 4) if diffs else 0.0,
        "ci95": [lo, hi],
        "wins": sum(d > 1e-9 for d in diffs),
        "ties": sum(abs(d) <= 1e-9 for d in diffs),
        "losses": sum(d < -1e-9 for d in diffs),
        "significant": bool(diffs) and (lo > 0 or hi < 0),
    }


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
    # Full-dataset validation is enforced; ``max_samples`` only subsamples afterwards (smoke runs).
    documents = validate_dataset(data)
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
    method_words: dict[str, list[int]] = {m: [] for m in methods}
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
        method_words["lead_3"].append(len(lead_summary.split()))

        # 2. YAKE Extractive
        yake_ext_res = extractive.summarize(doc_text)
        yake_ext_rouge = compute_rouge_metrics(
            summary=yake_ext_res.summary, reference=ref_summary
        )
        method_scores["yake_extractive"].append(yake_ext_rouge)
        method_words["yake_extractive"].append(len(yake_ext_res.summary.split()))

        # 3. LLM Unconditioned
        llm_uncond_res = abstractive.summarize(doc_text, use_keywords=False)
        llm_uncond_rouge = compute_rouge_metrics(
            summary=llm_uncond_res.summary, reference=ref_summary
        )
        method_scores["llm_unconditioned"].append(llm_uncond_rouge)
        method_words["llm_unconditioned"].append(len(llm_uncond_res.summary.split()))

        # 4. YAKE LLM Guided
        yake_abs_res = abstractive.summarize(doc_text, use_keywords=True)
        yake_abs_rouge = compute_rouge_metrics(
            summary=yake_abs_res.summary, reference=ref_summary
        )
        method_scores["yake_llm_guided"].append(yake_abs_rouge)
        method_words["yake_llm_guided"].append(len(yake_abs_res.summary.split()))

        # 5. Oracle LLM Guided
        oracle_res = abstractive.summarize(
            doc_text, external_keywords=gold_keywords, use_keywords=True
        )
        oracle_rouge = compute_rouge_metrics(
            summary=oracle_res.summary, reference=ref_summary
        )
        method_scores["oracle_llm_guided"].append(oracle_rouge)
        method_words["oracle_llm_guided"].append(len(oracle_res.summary.split()))

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

    ci = {
        m: {k: list(bootstrap_ci([s[k] for s in method_scores[m]])) for k in _METRICS}
        for m in methods
    }
    baseline = "lead_3"
    paired = {
        m: {
            k: paired_comparison([s[k] for s in method_scores[m]], [s[k] for s in method_scores[baseline]])
            for k in _METRICS
        }
        for m in methods
        if m != baseline
    }
    avg_words = {
        m: round(sum(method_words[m]) / len(method_words[m]), 1) if method_words[m] else 0.0
        for m in methods
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backend": backend,
        "model": model if backend != "mock" else None,
        "sample_ids": sample_ids,
        "reference_sources": {d["id"]: d["reference_source"] for d in documents},
        "methods": averaged_methods,
        "ci95": ci,
        "paired_vs_lead_3": paired,
        "avg_summary_words": avg_words,
        "avg_reference_words": round(
            sum(len(d["reference_summary"].split()) for d in documents) / len(documents), 1
        ),
        "raw_scores": method_scores,
    }


def _uncertainty_table(results: dict[str, Any]) -> str:
    rows = ["| Method | ROUGE-1 F1 [95% CI] | ROUGE-2 F1 [95% CI] | ROUGE-L F1 [95% CI] | Δ R1 vs Lead-3 (W/T/L) | Avg words |", "|---|---|---|---|---|---|"]
    for m, vals in results["methods"].items():
        cells = [f"{vals[k]:.3f} [{results['ci95'][m][k][0]:.3f}, {results['ci95'][m][k][1]:.3f}]" for k in _METRICS]
        if m == "lead_3":
            delta = "baseline"
        else:
            pc = results["paired_vs_lead_3"][m]["rouge1_f"]
            sig = " *" if pc["significant"] else ""
            delta = f"{pc['mean_diff']:+.3f}{sig} ({pc['wins']}/{pc['ties']}/{pc['losses']})"
        rows.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | {delta} | {results['avg_summary_words'][m]} |")
    rows.append("")
    rows.append("`*` = paired 95% CI excludes zero.")
    return "\n".join(rows)


def _provenance_line(results: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for src in results["reference_sources"].values():
        counts[src] = counts.get(src, 0) + 1
    return ", ".join(f"{v} × `{k}`" for k, v in sorted(counts.items()))


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

    model_part = f" (model `{results['model']}`)" if results.get("model") else ""
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
| **YAKE LLM Guided** | Full Pipeline | {m['yake_llm_guided']['rouge1_f']:.4f} | {m['yake_llm_guided']['rouge1_r']:.4f} | {m['yake_llm_guided']['rouge2_f']:.4f} | {m['yake_llm_guided']['rouge2_r']:.4f} | {m['yake_llm_guided']['rougeL_f']:.4f} | {m['yake_llm_guided']['rougeL_r']:.4f} |
| **Oracle LLM Guided** | Upper Bound (Gold KW) | {m['oracle_llm_guided']['rouge1_f']:.4f} | {m['oracle_llm_guided']['rouge1_r']:.4f} | {m['oracle_llm_guided']['rouge2_f']:.4f} | {m['oracle_llm_guided']['rouge2_r']:.4f} | {m['oracle_llm_guided']['rougeL_f']:.4f} | {m['oracle_llm_guided']['rougeL_r']:.4f} |

---

## 2. Uncertainty and length

With only {n_docs} short documents, differences between methods are often within noise. Values below are
mean ROUGE-1/2/L F1 with a 95% bootstrap CI, the paired difference against Lead-3 (wins/ties/losses per
document) and the average summary length (reference: {results['avg_reference_words']} words).

{_uncertainty_table(results)}

Reference provenance: {_provenance_line(results)}

---

## 3. Interpretation

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
