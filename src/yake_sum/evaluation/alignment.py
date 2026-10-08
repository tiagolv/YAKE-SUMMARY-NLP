"""Keyword/summary alignment metrics (reference-free coverage and gold-keyword P/R/F)."""

from __future__ import annotations

import re
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


_STOP = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its "
    "this that these those which who whom whose what when where why how not no nor than then so such "
    "can could may might will would shall should do does did done has have had having we our us they "
    "their them he she his her you your i me my also into over under between within without about "
    "more most less least very each both either neither other another any all some".split()
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def source_support(summary: str, source: str, max_listed: int = 10) -> dict[str, object]:
    """Cheap, deterministic faithfulness signal for generated summaries.

    * ``supported_ratio``: fraction of the summary's content words (stemmed, stopwords
      removed) that also occur in the source. Low values suggest paraphrase drift or
      invented content.
    * ``unsupported_numbers``: numbers in the summary that never appear in the source —
      the most damaging kind of hallucination in scientific text.
    * ``unsupported_terms``: a sample of content words absent from the source.

    This is a heuristic (a good paraphrase can legitimately use new words); it flags
    summaries to *review*, it does not prove faithfulness. Empty inputs yield 0.0.
    """
    if not summary.strip() or not source.strip():
        return {"supported_ratio": 0.0, "unsupported_numbers": [], "unsupported_terms": []}
    source_stems = set(stem_tokens(source))
    content = [
        (tok, st)
        for tok, st in zip(re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", summary.lower()), stem_tokens(summary))
        if tok not in _STOP and not tok.isdigit()
    ]
    if content:
        supported = sum(st in source_stems for _, st in content)
        ratio = supported / len(content)
        missing: list[str] = []
        for tok, st in content:
            if st not in source_stems and tok not in missing:
                missing.append(tok)
    else:
        ratio, missing = 0.0, []
    bad_numbers = sorted(_numbers(summary) - _numbers(source))
    return {
        "supported_ratio": round(ratio, 4),
        "unsupported_numbers": bad_numbers,
        "unsupported_terms": missing[:max_listed],
    }
