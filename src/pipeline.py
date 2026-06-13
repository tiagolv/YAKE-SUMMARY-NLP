from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_gold_keywords(path: Path, doc_id: str | None) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    if not documents:
        return []
    if doc_id:
        for item in documents:
            if item.get("id") == doc_id:
                return item.get("gold_keywords", [])
        # doc_id given but not found: no gold available for this document
        return []
    return documents[0].get("gold_keywords", [])


def run_pipeline(
    config_path: Path,
    input_path: Path,
    output_path: Path | None = None,
    gold_path: Path | None = None,
    reference_summary_path: Path | None = None,
    external_keywords: list[str] | None = None,
    ablation_mode: str | None = None,
    prompt_override: str | None = None,
) -> tuple[dict, Path]:
    """Run the YAKE + LLM pipeline.

    Args:
        config_path: Path to config YAML.
        input_path: Path to input text file.
        output_path: Path to save output JSON.
        gold_path: Path to gold keyword JSON.
        reference_summary_path: Path to reference summary text.
        external_keywords: If provided, use these instead of YAKE extraction.
        ablation_mode: One of None, 'no_keywords', 'keywords_only'.
        prompt_override: If provided, use this prompt string directly.
    """
    config = load_yaml(config_path)

    raw_text = load_text(input_path)
    clean_text = normalize_text(raw_text)

    yake_config = YakeConfig(**config["yake"])
    extractor = YakeKeywordExtractor(yake_config)

    # Decide keyword source
    if external_keywords is not None:
        keywords = external_keywords
        keywords_scored = [(kw, 0.0) for kw in keywords]
        keyword_source = "external"
    else:
        keywords_scored = extractor.extract(clean_text)
        keywords = [kw for kw, _ in keywords_scored]
        keyword_source = "yake"

    prompt_config = PromptConfig(**config["prompt"])
    truncated_text = truncate_text(clean_text, prompt_config.max_text_chars)

    # Build prompt based on mode
    if prompt_override:
        prompt = prompt_override
    elif ablation_mode == "no_keywords":
        prompt = build_prompt_no_keywords(truncated_text, prompt_config)
    elif ablation_mode == "keywords_only":
        prompt = build_prompt_keywords_only(keywords, prompt_config)
    else:
        prompt = build_prompt(keywords, truncated_text, prompt_config)

    llm_config = LLMConfig(**config["llm"])
    llm = create_llm_client(llm_config)
    summary = llm.generate(prompt)

    gold_keywords = None
    if gold_path and gold_path.exists():
        gold_keywords = load_gold_keywords(gold_path, input_path.stem)

    reference_summary = None
    if reference_summary_path and reference_summary_path.exists():
        reference_summary = load_text(reference_summary_path)

    metrics = evaluate_metrics(
        summary=summary,
        source_keywords=keywords,
        extractor=extractor,
        gold_keywords=gold_keywords,
        reference_summary=reference_summary,
    )

    result = {
        "input": str(input_path),
        "keyword_source": keyword_source,
        "ablation_mode": ablation_mode or "full",
        "keywords": [
            {"keyword": kw, "score": score} for kw, score in keywords_scored
        ],
        "summary": summary,
        "metrics": metrics,
        "config": {
            "yake": asdict(yake_config),
            "llm": asdict(llm_config),
            "prompt": asdict(prompt_config),
        },
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if output_path is None:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = outputs_dir / f"run_{timestamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAKE + LLM local pipeline")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", help="Path to output JSON")
    parser.add_argument("--gold", help="Path to gold keyword JSON")
    parser.add_argument("--reference-summary", help="Path to reference summary text")
    parser.add_argument(
        "--external-keywords",
        help="Comma-separated list of external keywords to use instead of YAKE",
    )
    parser.add_argument(
        "--ablation",
        choices=["no_keywords", "keywords_only"],
        help="Ablation mode: omit keywords or document text from prompt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else None
    gold_path = Path(args.gold) if args.gold else None
    reference_path = Path(args.reference_summary) if args.reference_summary else None
    ext_keywords = (
        [kw.strip() for kw in args.external_keywords.split(",")]
        if args.external_keywords
        else None
    )

    _, saved_path = run_pipeline(
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_path=output_path,
        gold_path=gold_path,
        reference_summary_path=reference_path,
        external_keywords=ext_keywords,
        ablation_mode=args.ablation,
    )

    print(f"Saved output to {saved_path}")


if __name__ == "__main__":
    main()
