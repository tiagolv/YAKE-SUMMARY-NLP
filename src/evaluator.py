"""Legacy evaluation helpers, now backed by ``yake_sum`` (stem-aware, zero-safe)."""

from __future__ import annotations

import re
from typing import Iterable

from yake_sum.evaluation.alignment import _contains
from yake_sum.evaluation.rouge import compute_rouge_metrics
from yake_sum.text import stem_tokens


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_coverage(summary: str, keywords: Iterable[str]) -> dict:
    """Which keywords appear in ``summary``.

    Matching is done on stemmed token sequences (so "neural networks" matches
    "neural network") and respects word boundaries (so "graph" no longer matches
    "paragraph", which the old substring check did).
    """
    summary_stems = stem_tokens(summary)
    keywords_list = list(keywords)
    found, missing = [], []
    for kw in keywords_list:
        stems = stem_tokens(kw)
        (found if stems and _contains(summary_stems, stems) else missing).append(kw)
    coverage = (len(found) / len(keywords_list)) if keywords_list else 0.0
    return {"coverage": coverage, "found": found, "missing": missing}


def keyword_precision_recall(predicted_keywords: Iterable[str], gold_keywords: Iterable[str]) -> dict:
    """Stem-aware set precision/recall between two keyword lists."""
    predicted = {" ".join(stem_tokens(k)): _normalize(k) for k in predicted_keywords if stem_tokens(k)}
    gold = {" ".join(stem_tokens(k)) for k in gold_keywords if stem_tokens(k)}
    hits = predicted.keys() & gold
    precision = len(hits) / len(predicted) if predicted else 0.0
    recall = len(hits) / len(gold) if gold else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positive": sorted(predicted[h] for h in hits),
    }


def compute_rouge(summary: str, reference: str) -> dict:
    """ROUGE F1 (stemmed). Keeps the legacy keys and adds ``rouge2_f``."""
    m = compute_rouge_metrics(summary=summary, reference=reference)
    return {"rouge1_f": m["rouge1_f"], "rouge2_f": m["rouge2_f"], "rougeL_f": m["rougeL_f"]}
