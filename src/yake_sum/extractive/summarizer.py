"""Extractive summarizer driven by YAKE keyword extraction."""

from __future__ import annotations

import re
from typing import Any

import yake

from ..models import ExtractiveResult, Keyword
from .scorer import YakeSentenceScorer


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly."""
    raw = text.strip()
    if not raw:
        return []
    # Split on sentence end punctuation followed by whitespace
    chunks = re.split(r"(?<=[.!?])\s+", raw)
    return [c.strip() for c in chunks if c.strip()]


class ExtractiveSummarizer:
    """Zero-LLM extractive summarizer based on YAKE keyword scoring."""

    def __init__(
        self,
        num_sentences: int = 3,
        language: str = "en",
        max_ngram_size: int = 3,
        top_k: int = 15,
        deduplication_threshold: float = 0.9,
    ) -> None:
        self.num_sentences = num_sentences
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.top_k = top_k
        self.deduplication_threshold = deduplication_threshold

        self._extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            top=self.top_k,
        )

    def summarize(self, text: str) -> ExtractiveResult:
        sentences = split_into_sentences(text)
        if not sentences:
            return ExtractiveResult(
                summary="",
                selected_sentences=[],
                keywords=[],
                sentence_scores=[],
                compression_ratio=0.0,
            )

        if len(sentences) <= self.num_sentences:
            kw_tuples = self._extractor.extract_keywords(text) if len(text.strip()) > 3 else []
            keywords = [Keyword(keyword=k, score=s) for k, s in kw_tuples]
            return ExtractiveResult(
                summary=" ".join(sentences),
                selected_sentences=sentences,
                keywords=keywords,
                sentence_scores=[1.0] * len(sentences),
                compression_ratio=0.0,
            )

        # Extract keywords
        kw_tuples = self._extractor.extract_keywords(text)
        keywords = [Keyword(keyword=k, score=round(s, 6)) for k, s in kw_tuples]

        # Score sentences
        scorer = YakeSentenceScorer(keywords_scored=kw_tuples)
        scores = scorer.score_sentences(sentences)

        # Pair each sentence with its original index and score
        indexed_scores = list(enumerate(scores))
        # Sort descending by score
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top num_sentences
        top_picks = indexed_scores[: self.num_sentences]
        # Re-sort chronologically by original index
        top_picks.sort(key=lambda x: x[0])

        selected_sentences = [sentences[idx] for idx, _ in top_picks]
        summary = " ".join(selected_sentences)

        source_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = (
            round(1.0 - (summary_words / source_words), 4) if source_words > 0 else 0.0
        )

        return ExtractiveResult(
            summary=summary,
            selected_sentences=selected_sentences,
            keywords=keywords,
            sentence_scores=scores,
            compression_ratio=compression_ratio,
        )
