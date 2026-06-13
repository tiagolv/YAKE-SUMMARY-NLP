"""Prompt comparison — evaluate different prompt strategies.

Runs all prompt templates over the same documents and compares
keyword coverage, summary alignment, and output characteristics.

Usage:
    python -m src.prompt_comparison --config config.yaml
    python -m src.prompt_comparison --config config.yaml --output outputs/prompt_comparison.json
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
from .prompt_templates import TEMPLATES, build_from_template


def run_prompt_experiment(
    doc_path: Path,
    base_config: dict,
    template_name: str,
    gold_path: Path | None = None,
) -> dict:
    """Run a single prompt template experiment."""
    raw_text = doc_path.read_text(encoding="utf-8")
    clean_text = normalize_text(raw_text)

    yake_config = YakeConfig(**base_config["yake"])
    extractor = YakeKeywordExtractor(yake_config)
    keywords_scored = extractor.extract(clean_text)
    keywords = [kw for kw, _ in keywords_scored]

    max_chars = base_config["prompt"].get("max_text_chars", 3000)
    truncated_text = truncate_text(clean_text, max_chars)

    # Build prompt from template
    template = TEMPLATES[template_name]
    prompt = build_from_template(template, keywords, truncated_text)

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
    )

    return {
        "document": doc_path.stem,
        "template": template_name,
        "template_description": template.description,
        "keywords": keywords,
        "summary": summary,
        "summary_length": len(summary.split()),
        "metrics": {
            "coverage": metrics["keyword_coverage"]["coverage"],
            "found_keywords": len(metrics["keyword_coverage"]["found"]),
            "missing_keywords": len(metrics["keyword_coverage"]["missing"]),
            "alignment_precision": metrics["summary_alignment"]["precision"],
            "alignment_recall": metrics["summary_alignment"]["recall"],
        },
    }


def generate_comparison_table(results: list[dict]) -> str:
    """Generate a markdown comparison table for prompt experiments."""
    header = (
        "| Document | Template | Summary Words | Coverage | "
        "Align Prec | Align Recall |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        m = r["metrics"]
        row = (
            f'| {r["document"][:25]} | {r["template"]} | '
            f'{r["summary_length"]} | {m["coverage"]:.3f} | '
            f'{m["alignment_precision"]:.3f} | {m["alignment_recall"]:.3f} |'
        )
        rows.append(row)
    return header + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt strategy comparison")
    parser.add_argument("--config", default="config.yaml", help="Base config YAML")
    parser.add_argument("--samples-dir", default="data/samples", help="Input texts directory")
    parser.add_argument("--gold", default="data/eval/inspec_gold.json", help="Gold keywords JSON")
    parser.add_argument("--output", default="outputs/prompt_comparison.json", help="Output JSON")
    parser.add_argument("--docs", type=int, default=0, help="Limit number of documents (0 = all)")
    args = parser.parse_args()

    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    documents = sorted(Path(args.samples_dir).glob("*.txt"))
    if args.docs > 0:
        documents = documents[: args.docs]
    gold_path = Path(args.gold)

    template_names = list(TEMPLATES.keys())
    total = len(documents) * len(template_names)
    print(f"Comparing {len(template_names)} prompt templates × {len(documents)} docs = {total} runs")

    results = []
    for doc in documents:
        print(f"\n  Document: {doc.stem}")
        for tname in template_names:
            print(f"    Template: {tname}...", end=" ", flush=True)
            try:
                result = run_prompt_experiment(doc, base_config, tname, gold_path)
                results.append(result)
                print(f"coverage={result['metrics']['coverage']:.3f}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "document": doc.stem,
                    "template": tname,
                    "error": str(exc),
                })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "prompt_comparison",
        "templates": {k: v.description for k, v in TEMPLATES.items()},
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
        table = generate_comparison_table(successful)

        # Per-template averages
        avg_section = "\n\n## Average Metrics by Template\n\n"
        avg_section += "| Template | Description | Avg Coverage | Avg Align Prec | Avg Align Recall | Avg Words |\n"
        avg_section += "|---|---|---|---|---|---|\n"
        for tname in template_names:
            tresults = [r for r in successful if r["template"] == tname]
            if tresults:
                avg_cov = sum(r["metrics"]["coverage"] for r in tresults) / len(tresults)
                avg_prec = sum(r["metrics"]["alignment_precision"] for r in tresults) / len(tresults)
                avg_rec = sum(r["metrics"]["alignment_recall"] for r in tresults) / len(tresults)
                avg_words = sum(r["summary_length"] for r in tresults) / len(tresults)
                desc = TEMPLATES[tname].description[:40]
                avg_section += (
                    f"| {tname} | {desc} | {avg_cov:.3f} | "
                    f"{avg_prec:.3f} | {avg_rec:.3f} | {avg_words:.0f} |\n"
                )

        table_path.write_text(
            f"# Prompt Strategy Comparison\n\n{table}\n{avg_section}",
            encoding="utf-8",
        )
        print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
