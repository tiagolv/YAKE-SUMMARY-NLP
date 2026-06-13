from __future__ import annotations

from dataclasses import dataclass

try:
    import yake
except ImportError as exc:
    raise ImportError("Missing dependency 'yake'. Install it with: pip install yake") from exc


@dataclass
class YakeConfig:
    language: str = "en"
    max_ngram_size: int = 3
    deduplication_threshold: float = 0.9
    top_k: int = 12


class YakeKeywordExtractor:
    def __init__(self, config: YakeConfig) -> None:
        self._extractor = yake.KeywordExtractor(
            lan=config.language,
            n=config.max_ngram_size,
            dedupLim=config.deduplication_threshold,
            top=config.top_k,
        )

    def extract(self, text: str) -> list[tuple[str, float]]:
        return self._extractor.extract_keywords(text)
