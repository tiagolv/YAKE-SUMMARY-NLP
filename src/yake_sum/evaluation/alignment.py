"""Keyword/summary alignment metrics (reference-free coverage and gold-keyword P/R/F)."""

from __future__ import annotations

from typing import Iterable

from ..text import stem_tokens


def _contains(seq: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    n, m = len(seq), len(sub)
    return 0 < m <= n and any(seq[i : i + m] == sub for i in range(n - m + 1))


def keyword_coverage(summary: str, keywords: Iterable[str]) -> float:
    """Fraction of distinct keywords (stem-aware phrase match) present in ``summary``.

    Returns 0.0 for an empty summary or no keywords (never divides by zero).
    """
    stems = {s for s in (stem_tokens(k) for k in keywords) if s}
    if not stems or not summary.strip():
        return 0.0
    seq = stem_tokens(summary)
    return round(sum(_contains(seq, s) for s in stems) / len(stems), 4)


def keyword_prf(predicted: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    """Precision/recall/F1 of predicted vs. gold keywords using stemmed exact match."""
    pred = {s for s in (stem_tokens(k) for k in predicted) if s}
    ref = {s for s in (stem_tokens(k) for k in gold) if s}
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(pred & ref)
    p, r = hit / len(pred), hit / len(ref)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
