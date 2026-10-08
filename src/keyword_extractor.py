from __future__ import annotations

from dataclasses import dataclass

from yake_sum.keywords import KeywordExtractor


@dataclass
class YakeConfig:
    language: str = "en"
    max_ngram_size: int = 3
    deduplication_threshold: float = 0.9
    top_k: int = 12


class YakeKeywordExtractor:
    """Legacy facade over :class:`yake_sum.keywords.KeywordExtractor`.

    Same interface as before, but empty / degenerate text now yields ``[]`` instead
    of raising.
    """

    def __init__(self, config: YakeConfig) -> None:
        self._extractor = KeywordExtractor(
            language=config.language,
            max_ngram_size=config.max_ngram_size,
            top_k=config.top_k,
            deduplication_threshold=config.deduplication_threshold,
        )

    def extract(self, text: str) -> list[tuple[str, float]]:
        return self._extractor.extract(text)
