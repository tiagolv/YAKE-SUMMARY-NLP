from __future__ import annotations

from typing import Iterable

from .evaluator import compute_rouge, keyword_coverage, keyword_precision_recall
from .keyword_extractor import YakeKeywordExtractor
from .preprocessor import normalize_text


def extract_keywords_for_alignment(text: str, extractor: YakeKeywordExtractor) -> list[str]:
    clean_text = normalize_text(text)
    keywords_scored = extractor.extract(clean_text)
    return [keyword for keyword, _ in keywords_scored]


def evaluate_metrics(
    summary: str,
    source_keywords: Iterable[str],
    extractor: YakeKeywordExtractor,
    gold_keywords: Iterable[str] | None = None,
    reference_summary: str | None = None,
) -> dict:
    source_keywords_list = list(source_keywords)
    metrics = {"keyword_coverage": keyword_coverage(summary, source_keywords_list)}

    summary_keywords = extract_keywords_for_alignment(summary, extractor)
    alignment = keyword_precision_recall(summary_keywords, source_keywords_list)
    alignment["summary_keywords"] = summary_keywords
    metrics["summary_alignment"] = alignment

    if gold_keywords:
        metrics["yake_vs_gold"] = keyword_precision_recall(source_keywords_list, gold_keywords)

    if reference_summary:
        metrics["rouge"] = compute_rouge(summary, reference_summary)

    return metrics
