"""Safe YAKE wrapper shared by all engines."""

from __future__ import annotations

import yake

from .validation import require_float, require_int


class KeywordExtractor:
    """Thin, failure-tolerant wrapper around ``yake.KeywordExtractor``.

    Returns ``[]`` (instead of raising) for empty, stopword-only or too-short text,
    so callers can fall back to unconditioned behaviour.
    """

    def __init__(
        self,
        language: str = "en",
        max_ngram_size: int = 3,
        top_k: int = 15,
        deduplication_threshold: float = 0.9,
    ) -> None:
        self.language = language
        self.max_ngram_size = require_int("max_ngram_size", max_ngram_size)
        self.top_k = require_int("top_k", top_k)
        self.deduplication_threshold = require_float(
            "deduplication_threshold", deduplication_threshold, 0.0, 1.0
        )
        self._extractor = yake.KeywordExtractor(
            lan=language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            top=self.top_k,
        )

    def extract(self, text: str) -> list[tuple[str, float]]:
        clean = text.strip()
        if len(clean.split()) < 2:
            return []
        try:
            pairs = self._extractor.extract_keywords(clean)
        except Exception:  # YAKE can fail on degenerate input; treat as "no keywords"
            return []
        return [(k, float(s)) for k, s in pairs if k and k.strip()]
