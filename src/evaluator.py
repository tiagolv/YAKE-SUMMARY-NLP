from __future__ import annotations

import re
from typing import Iterable


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_coverage(summary: str, keywords: Iterable[str]) -> dict:
    summary_norm = _normalize(summary)
    keywords_list = list(keywords)
    found = []
    missing = []
    for kw in keywords_list:
        kw_norm = _normalize(kw)
        if kw_norm and kw_norm in summary_norm:
            found.append(kw)
        else:
            missing.append(kw)
    coverage = (len(found) / len(keywords_list)) if keywords_list else 0.0
    return {"coverage": coverage, "found": found, "missing": missing}


def keyword_precision_recall(predicted_keywords: Iterable[str], gold_keywords: Iterable[str]) -> dict:
    predicted_norm = {_normalize(k) for k in predicted_keywords if _normalize(k)}
    gold_norm = {_normalize(k) for k in gold_keywords if _normalize(k)}
    true_positive = predicted_norm & gold_norm

    precision = len(true_positive) / len(predicted_norm) if predicted_norm else 0.0
    recall = len(true_positive) / len(gold_norm) if gold_norm else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "true_positive": sorted(true_positive),
    }


def compute_rouge(summary: str, reference: str) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("Missing dependency 'rouge-score'. Install it with: pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }
