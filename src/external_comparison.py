"""External keywords comparison — compare YAKE local vs. external keyword sources.

Runs the pipeline twice per document: once with YAKE-extracted keywords
and once with externally provided keywords, then compares the generated summaries.

Usage:
    python -m src.external_comparison --config config.yaml
    python -m src.external_comparison --config config.yaml --external data/external_keywords
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .evaluation import evaluate_metrics
from .evaluator import compute_rouge
from .external_keywords import load_all_external_keywords
from .keyword_extractor import YakeConfig, YakeKeywordExtractor
from .llm_interface import LLMConfig, create_llm_client
from .preprocessor import normalize_text, truncate_text
from .prompt_builder import PromptConfig, build_prompt


def run_comparison(
    doc_path: Path,
    base_config: dict,
    external_keywords: list[str],
    gold_path: Path | None = None,
) -> dict:
    """Run the pipeline with YAKE and with external keywords, then compare."""
    raw_text = doc_path.read_text(encoding="utf-8")
    clean_text = normalize_text(raw_text)

    yake_config = YakeConfig(**base_config["yake"])
    extractor = YakeKeywordExtractor(yake_config)
    yake_keywords_scored = extractor.extract(clean_text)
    yake_keywords = [kw for kw, _ in yake_keywords_scored]

    prompt_config = PromptConfig(**base_config["prompt"])
    truncated_text = truncate_text(clean_text, prompt_config.max_text_chars)

    llm_config = LLMConfig(**base_config["llm"])
    llm = create_llm_client(llm_config)

    # Run with YAKE keywords
    prompt_yake = build_prompt(yake_keywords, truncated_text, prompt_config)
    summary_yake = llm.generate(prompt_yake)

    # Run with external keywords
    prompt_ext = build_prompt(external_keywords, truncated_text, prompt_config)
    summary_ext = llm.generate(prompt_ext)

    # Evaluate both
    gold_keywords = None
    if gold_path and gold_path.exists():
        gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
        for item in gold_data.get("documents", []):
            if item.get("id") == doc_path.stem:
                gold_keywords = item.get("gold_keywords", [])
                break

    metrics_yake = evaluate_metrics(
        summary=summary_yake,
        source_keywords=yake_keywords,
        extractor=extractor,
        gold_keywords=gold_keywords,
    )

    metrics_ext = evaluate_metrics(
        summary=summary_ext,
        source_keywords=external_keywords,
        extractor=extractor,
        gold_keywords=gold_keywords,
    )

    # Compare summaries via ROUGE
    cross_rouge = compute_rouge(summary_yake, summary_ext)

    # Keyword overlap between YAKE and external
    yake_set = {kw.lower() for kw in yake_keywords}
    ext_set = {kw.lower() for kw in external_keywords}
    overlap = yake_set & ext_set
    kw_jaccard = len(overlap) / len(yake_set | ext_set) if (yake_set | ext_set) else 0.0

    return {
        "document": doc_path.stem,
        "yake_keywords": yake_keywords,
        "external_keywords": external_keywords,
        "keyword_overlap": {
            "common": sorted(overlap),
            "yake_only": sorted(yake_set - ext_set),
            "external_only": sorted(ext_set - yake_set),
            "jaccard_similarity": kw_jaccard,
        },
        "yake_summary": {
            "text": summary_yake,
            "word_count": len(summary_yake.split()),
            "coverage": metrics_yake["keyword_coverage"]["coverage"],
            "alignment_precision": metrics_yake["summary_alignment"]["precision"],
        },
        "external_summary": {
            "text": summary_ext,
            "word_count": len(summary_ext.split()),
            "coverage": metrics_ext["keyword_coverage"]["coverage"],
            "alignment_precision": metrics_ext["summary_alignment"]["precision"],
        },
        "cross_comparison": {
            "rouge1_between_summaries": cross_rouge["rouge1_f"],
            "rougeL_between_summaries": cross_rouge["rougeL_f"],
        },
    }


def generate_comparison_table(results: list[dict]) -> str:
    """Generate a comparison table."""
    header = (
        "| Document | KW Jaccard | YAKE Cov | Ext Cov | "
        "YAKE Prec | Ext Prec | ROUGE-1 (cross) | ROUGE-L (cross) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        y = r["yake_summary"]
        e = r["external_summary"]
        c = r["cross_comparison"]
        row = (
            f'| {r["document"][:25]} | '
            f'{r["keyword_overlap"]["jaccard_similarity"]:.3f} | '
            f'{y["coverage"]:.3f} | {e["coverage"]:.3f} | '
            f'{y["alignment_precision"]:.3f} | {e["alignment_precision"]:.3f} | '
            f'{c["rouge1_between_summaries"]:.3f} | '
            f'{c["rougeL_between_summaries"]:.3f} |'
        )
        rows.append(row)
    return header + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="External keywords comparison")
    parser.add_argument("--config", default="config.yaml", help="Base config YAML")
    parser.add_argument("--samples-dir", default="data/samples", help="Input texts directory")
    parser.add_argument(
        "--external",
        default="data/external_keywords",
        help="External keywords directory or JSON file",
    )
    parser.add_argument("--gold", default="data/eval/inspec_gold.json", help="Gold keywords JSON")
    parser.add_argument("--output", default="outputs/external_comparison.json", help="Output JSON")
    args = parser.parse_args()

    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ext_data = load_all_external_keywords(Path(args.external))
    documents = sorted(Path(args.samples_dir).glob("*.txt"))
    gold_path = Path(args.gold)

    # Only process documents that have external keywords
    available_docs = [d for d in documents if d.stem in ext_data]
    print(f"Comparing YAKE vs. external keywords for {len(available_docs)} documents...")

    results = []
    for doc in available_docs:
        print(f"  {doc.stem}...", end=" ", flush=True)
        try:
            result = run_comparison(
                doc_path=doc,
                base_config=base_config,
                external_keywords=ext_data[doc.stem],
                gold_path=gold_path,
            )
            results.append(result)
            print(f"jaccard={result['keyword_overlap']['jaccard_similarity']:.3f}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "external_keywords_comparison",
        "external_source": str(args.external),
        "total_documents": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")

    if results:
        table_path = output_path.with_suffix(".md")
        table = generate_comparison_table(results)

        # Averages
        avg_section = "\n\n## Summary\n\n"
        avg_jaccard = sum(r["keyword_overlap"]["jaccard_similarity"] for r in results) / len(results)
        avg_yake_cov = sum(r["yake_summary"]["coverage"] for r in results) / len(results)
        avg_ext_cov = sum(r["external_summary"]["coverage"] for r in results) / len(results)
        avg_cross_r1 = sum(r["cross_comparison"]["rouge1_between_summaries"] for r in results) / len(results)

        avg_section += f"- **Average keyword Jaccard similarity**: {avg_jaccard:.3f}\n"
        avg_section += f"- **Average YAKE coverage**: {avg_yake_cov:.3f}\n"
        avg_section += f"- **Average external coverage**: {avg_ext_cov:.3f}\n"
        avg_section += f"- **Average cross-summary ROUGE-1**: {avg_cross_r1:.3f}\n"

        table_path.write_text(
            f"# External Keywords Comparison: YAKE Local vs. YAKE Online Demo\n\n{table}\n{avg_section}",
            encoding="utf-8",
        )
        print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
