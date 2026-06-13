"""Batch runner — systematic experimentation with parameter variation.

Runs the pipeline over all documents in data/samples/ with multiple
parameter combinations and generates a consolidated results report.

Usage:
    python -m src.batch_runner --config config.yaml
    python -m src.batch_runner --config config.yaml --output outputs/batch_results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import yaml

from .evaluation import evaluate_metrics
from .keyword_extractor import YakeConfig, YakeKeywordExtractor
from .llm_interface import LLMConfig, create_llm_client
from .preprocessor import normalize_text, truncate_text
from .prompt_builder import PromptConfig, build_prompt


# Parameter grid for systematic experimentation
PARAM_GRID = {
    "top_k": [5, 10, 15, 20],
    "max_ngram_size": [1, 2, 3],
    "temperature": [0.1, 0.3, 0.7],
}


def collect_documents(samples_dir: Path) -> list[Path]:
    """Collect all .txt files from the samples directory."""
    docs = sorted(samples_dir.glob("*.txt"))
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {samples_dir}")
    return docs


def run_single_experiment(
    doc_path: Path,
    base_config: dict,
    top_k: int,
    max_ngram_size: int,
    temperature: float,
    gold_path: Path | None = None,
) -> dict:
    """Run a single experiment with specific parameters."""
    raw_text = doc_path.read_text(encoding="utf-8")
    clean_text = normalize_text(raw_text)

    # Override YAKE parameters
    yake_params = {**base_config["yake"], "top_k": top_k, "max_ngram_size": max_ngram_size}
    yake_config = YakeConfig(**yake_params)
    extractor = YakeKeywordExtractor(yake_config)
    keywords_scored = extractor.extract(clean_text)
    keywords = [kw for kw, _ in keywords_scored]

    # Build prompt
    prompt_config = PromptConfig(**base_config["prompt"])
    truncated_text = truncate_text(clean_text, prompt_config.max_text_chars)
    prompt = build_prompt(keywords, truncated_text, prompt_config)

    # Override LLM temperature
    llm_params = {**base_config["llm"], "temperature": temperature}
    llm_config = LLMConfig(**llm_params)
    llm = create_llm_client(llm_config)
    summary = llm.generate(prompt)

    # Load gold keywords if available
    gold_keywords = None
    if gold_path and gold_path.exists():
        gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
        for item in gold_data.get("documents", []):
            if item.get("id") == doc_path.stem:
                gold_keywords = item.get("gold_keywords", [])
                break

    # Evaluate
    metrics = evaluate_metrics(
        summary=summary,
        source_keywords=keywords,
        extractor=extractor,
        gold_keywords=gold_keywords,
    )

    return {
        "document": doc_path.stem,
        "params": {
            "top_k": top_k,
            "max_ngram_size": max_ngram_size,
            "temperature": temperature,
        },
        "num_keywords": len(keywords),
        "keywords": keywords,
        "summary": summary,
        "metrics": {
            "coverage": metrics["keyword_coverage"]["coverage"],
            "alignment_precision": metrics["summary_alignment"]["precision"],
            "alignment_recall": metrics["summary_alignment"]["recall"],
            "yake_vs_gold_precision": (
                metrics.get("yake_vs_gold", {}).get("precision")
            ),
            "yake_vs_gold_recall": (
                metrics.get("yake_vs_gold", {}).get("recall")
            ),
        },
    }


def generate_markdown_table(results: list[dict]) -> str:
    """Generate a markdown table from batch results."""
    header = (
        "| Document | top_k | n-gram | Temp | Keywords | Coverage | "
        "Align Prec | Align Recall | Gold Prec | Gold Recall |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        p = r["params"]
        m = r["metrics"]
        gp = f'{m["yake_vs_gold_precision"]:.3f}' if m["yake_vs_gold_precision"] is not None else "—"
        gr = f'{m["yake_vs_gold_recall"]:.3f}' if m["yake_vs_gold_recall"] is not None else "—"
        row = (
            f'| {r["document"][:25]} | {p["top_k"]} | {p["max_ngram_size"]} | '
            f'{p["temperature"]} | {r["num_keywords"]} | '
            f'{m["coverage"]:.3f} | {m["alignment_precision"]:.3f} | '
            f'{m["alignment_recall"]:.3f} | {gp} | {gr} |'
        )
        rows.append(row)
    return header + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for parameter variation")
    parser.add_argument("--config", default="config.yaml", help="Base config YAML")
    parser.add_argument("--samples-dir", default="data/samples", help="Directory with input texts")
    parser.add_argument("--gold", default="data/eval/inspec_gold.json", help="Gold keywords JSON")
    parser.add_argument("--output", default="outputs/batch_results.json", help="Output JSON path")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only test 2 documents with reduced grid",
    )
    args = parser.parse_args()

    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    documents = collect_documents(Path(args.samples_dir))
    gold_path = Path(args.gold) if args.gold else None

    # In quick mode, reduce the grid
    if args.quick:
        documents = documents[:2]
        param_combos = [(10, 3, 0.2)]
    else:
        param_combos = list(
            product(
                PARAM_GRID["top_k"],
                PARAM_GRID["max_ngram_size"],
                PARAM_GRID["temperature"],
            )
        )

    total = len(documents) * len(param_combos)
    print(f"Running {total} experiments ({len(documents)} docs × {len(param_combos)} configs)...")

    results = []
    for i, (doc, (top_k, ngram, temp)) in enumerate(
        product(documents, param_combos), 1
    ):
        print(f"  [{i}/{total}] {doc.stem} | top_k={top_k} ngram={ngram} temp={temp}")
        try:
            result = run_single_experiment(
                doc_path=doc,
                base_config=base_config,
                top_k=top_k,
                max_ngram_size=ngram,
                temperature=temp,
                gold_path=gold_path,
            )
            results.append(result)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append({
                "document": doc.stem,
                "params": {"top_k": top_k, "max_ngram_size": ngram, "temperature": temp},
                "error": str(exc),
            })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "batch_parameter_variation",
        "total_runs": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "param_grid": PARAM_GRID,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} results to {output_path}")

    # Generate markdown table
    successful = [r for r in results if "error" not in r]
    if successful:
        table_path = output_path.with_suffix(".md")
        table = generate_markdown_table(successful)
        table_path.write_text(
            f"# Batch Results — Parameter Variation\n\n{table}\n",
            encoding="utf-8",
        )
        print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
