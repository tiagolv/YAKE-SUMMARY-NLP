"""Ablation study — measure the contribution of each pipeline component.

Runs three conditions per document:
  1. full:          keywords + document text (standard pipeline)
  2. no_keywords:   document text only (no keyword guidance)
  3. keywords_only: keywords only (no document text)

Usage:
    python -m src.ablation_runner --config config.yaml
    python -m src.ablation_runner --config config.yaml --output outputs/ablation_results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .evaluation import evaluate_metrics
from .keyword_extractor import YakeConfig, YakeKeywordExtractor
from .llm_interface import LLMConfig, create_llm_client
from .preprocessor import normalize_text, truncate_text
from .prompt_builder import (
    PromptConfig,
    build_prompt,
    build_prompt_keywords_only,
    build_prompt_no_keywords,
)

ABLATION_MODES = ["full", "no_keywords", "keywords_only"]


def run_ablation_experiment(
    doc_path: Path,
    base_config: dict,
    mode: str,
    gold_path: Path | None = None,
) -> dict:
    """Run a single ablation experiment."""
    raw_text = doc_path.read_text(encoding="utf-8")
    clean_text = normalize_text(raw_text)

    yake_config = YakeConfig(**base_config["yake"])
    extractor = YakeKeywordExtractor(yake_config)
    keywords_scored = extractor.extract(clean_text)
    keywords = [kw for kw, _ in keywords_scored]

    prompt_config = PromptConfig(**base_config["prompt"])
    truncated_text = truncate_text(clean_text, prompt_config.max_text_chars)

    # Build prompt based on ablation mode
    if mode == "no_keywords":
        prompt = build_prompt_no_keywords(truncated_text, prompt_config)
    elif mode == "keywords_only":
        prompt = build_prompt_keywords_only(keywords, prompt_config)
    else:  # full
        prompt = build_prompt(keywords, truncated_text, prompt_config)

    llm_config = LLMConfig(**base_config["llm"])
    llm = create_llm_client(llm_config)
    summary = llm.generate(prompt)

    # Load gold keywords
    gold_keywords = None
    if gold_path and gold_path.exists():
        gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
        for item in gold_data.get("documents", []):
            if item.get("id") == doc_path.stem:
                gold_keywords = item.get("gold_keywords", [])
                break

    metrics = evaluate_metrics(
        summary=summary,
        source_keywords=keywords,
        extractor=extractor,
        gold_keywords=gold_keywords,
        source_text=clean_text,          # <-- adicionado
    )

    return {
        "document": doc_path.stem,
        "mode": mode,
        "keywords": keywords,
        "summary": summary,
        "summary_length": len(summary.split()),
        "metrics": {
            "compression_ratio": metrics.get("compression_ratio"),
            "coverage": metrics["keyword_coverage"]["coverage"],
            "found_keywords": len(metrics["keyword_coverage"]["found"]),
            "missing_keywords": len(metrics["keyword_coverage"]["missing"]),
            "alignment_precision": metrics["summary_alignment"]["precision"],
            "alignment_recall": metrics["summary_alignment"]["recall"],
        },
    }


def generate_ablation_table(results: list[dict]) -> str:
    """Generate a markdown comparison table for ablation results."""
    header = (
        "| Document | Mode | Summary Words | Compression | Coverage | "
        "Found KW | Missing KW | Align Prec | Align Recall |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        m = r["metrics"]
        row = (
            f'| {r["document"][:25]} | {r["mode"]} | {r["summary_length"]} | '
            f'{m["compression_ratio"]:.3f} | {m["coverage"]:.3f} | '
            f'{m["found_keywords"]} | {m["missing_keywords"]} | '
            f'{m["alignment_precision"]:.3f} | {m["alignment_recall"]:.3f} |'
        )
        rows.append(row)
    return header + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--config", default="config.yaml", help="Base config YAML")
    parser.add_argument("--samples-dir", default="data/samples", help="Input texts directory")
    parser.add_argument("--gold", default="data/eval/inspec_gold.json", help="Gold keywords JSON")
    parser.add_argument("--output", default="outputs/ablation_results.json", help="Output JSON")
    parser.add_argument("--docs", type=int, default=0, help="Limit number of documents (0 = all)")
    args = parser.parse_args()

    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    documents = sorted(Path(args.samples_dir).glob("*.txt"))
    if args.docs > 0:
        documents = documents[: args.docs]
    gold_path = Path(args.gold)

    total = len(documents) * len(ABLATION_MODES)
    print(f"Running ablation study: {len(documents)} docs × {len(ABLATION_MODES)} modes = {total} runs")

    results = []
    for doc in documents:
        print(f"\n  Document: {doc.stem}")
        for mode in ABLATION_MODES:
            print(f"    Mode: {mode}...", end=" ", flush=True)
            try:
                result = run_ablation_experiment(doc, base_config, mode, gold_path)
                results.append(result)
                print(f"coverage={result['metrics']['coverage']:.3f}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "document": doc.stem,
                    "mode": mode,
                    "error": str(exc),
                })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "ablation_study",
        "modes": ABLATION_MODES,
        "total_runs": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")

    # Generate table
    successful = [r for r in results if "error" not in r]
    if successful:
        table_path = output_path.with_suffix(".md")
        table = generate_ablation_table(successful)

        # Also compute per-mode averages
        avg_section = "\n\n## Average Metrics by Mode\n\n"
        avg_section += "| Mode | Avg Compression | Avg Coverage | Avg Align Prec | Avg Align Recall | Avg Words |\n"
        avg_section += "|---|---|---|---|---|---|\n"
        for mode in ABLATION_MODES:
            mode_results = [r for r in successful if r["mode"] == mode]
            if mode_results:
                avg_comp = sum(r["metrics"]["compression_ratio"] for r in mode_results) / len(mode_results)
                avg_cov = sum(r["metrics"]["coverage"] for r in mode_results) / len(mode_results)
                avg_prec = sum(r["metrics"]["alignment_precision"] for r in mode_results) / len(mode_results)
                avg_rec = sum(r["metrics"]["alignment_recall"] for r in mode_results) / len(mode_results)
                avg_words = sum(r["summary_length"] for r in mode_results) / len(mode_results)
                avg_section += f"| {mode} | {avg_comp:.3f} | {avg_cov:.3f} | {avg_prec:.3f} | {avg_rec:.3f} | {avg_words:.0f} |\n"

        table_path.write_text(
            f"# Ablation Study Results\n\n{table}\n{avg_section}",
            encoding="utf-8",
        )
        print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()