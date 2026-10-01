"""Extractive summarizer driven by YAKE keyword extraction (zero LLM, CPU only)."""

from __future__ import annotations

from ..keywords import KeywordExtractor
from ..models import ExtractiveResult, Keyword
from ..text import split_into_sentences
from ..validation import require_int
from .scorer import YakeSentenceScorer

__all__ = ["ExtractiveSummarizer", "split_into_sentences"]


class ExtractiveSummarizer:
    """Selects the ``num_sentences`` most keyword-dense, non-redundant sentences."""

    def __init__(
        self,
        num_sentences: int = 3,
        language: str = "en",
        max_ngram_size: int = 3,
        top_k: int = 15,
        deduplication_threshold: float = 0.9,
        redundancy_decay: float = 0.0,
        weight_power: float = 1.0,
    ) -> None:
        self.num_sentences = require_int("num_sentences", num_sentences)
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.top_k = top_k
        self.deduplication_threshold = deduplication_threshold
        self.redundancy_decay = redundancy_decay
        self.weight_power = weight_power
        self._extractor = KeywordExtractor(
            language=language,
            max_ngram_size=max_ngram_size,
            top_k=top_k,
            deduplication_threshold=deduplication_threshold,
        )

    def summarize(self, text: str) -> ExtractiveResult:
        sentences = split_into_sentences(text)
        if not sentences:
            return ExtractiveResult("", [], [], [], 0.0)

        kw_tuples = self._extractor.extract(text)
        keywords = [Keyword(keyword=k, score=round(s, 6)) for k, s in kw_tuples]
        scorer = YakeSentenceScorer(
            kw_tuples,
            redundancy_decay=self.redundancy_decay,
            weight_power=self.weight_power,
        )
        scores = scorer.score_sentences(sentences)

        if len(sentences) <= self.num_sentences:
            picks = list(range(len(sentences)))
        else:
            picks = scorer.select(sentences, max_units=self.num_sentences)

        selected = [sentences[i] for i in picks]
        summary = " ".join(selected)
        source_words = len(text.split())
        ratio = round(1.0 - len(summary.split()) / source_words, 4) if source_words else 0.0
        return ExtractiveResult(
            summary=summary,
            selected_sentences=selected,
            keywords=keywords,
            sentence_scores=scores,
            compression_ratio=max(ratio, 0.0),
            metadata={
                "fallback": "lead" if not scorer.has_keywords else None,
                "selected_indices": picks,
            },
        )
