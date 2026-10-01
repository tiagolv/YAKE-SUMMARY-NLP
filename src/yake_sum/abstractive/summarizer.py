"""Abstractive summarizer using YAKE keywords and LLM inference."""

from __future__ import annotations

from ..backends.base import BackendError, BaseLLMClient
from ..backends.mock import MockLLMClient
from ..keywords import KeywordExtractor
from ..models import AbstractiveResult, Keyword
from ..validation import require_float, require_int
from .prompt import build_keyword_prompt, build_unconditioned_prompt


class AbstractiveSummarizer:
    """Keyword-guided abstractive summarizer.

    Falls back to the unconditioned prompt when YAKE yields no keywords, and
    raises ``BackendError`` if the model returns an empty summary (never silently
    returns an empty string as if it were a valid result).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        language: str = "en",
        max_ngram_size: int = 3,
        top_k: int = 12,
        deduplication_threshold: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_sentences: int | None = 3,
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.temperature = require_float("temperature", temperature, 0.0, 2.0)
        self.max_tokens = require_int("max_tokens", max_tokens)
        self.max_sentences = None if max_sentences is None else require_int("max_sentences", max_sentences)
        self._extractor = KeywordExtractor(language, max_ngram_size, top_k, deduplication_threshold)

    def summarize(
        self,
        text: str,
        external_keywords: list[str] | None = None,
        use_keywords: bool = True,
    ) -> AbstractiveResult:
        clean = text.strip()
        if not clean:
            return AbstractiveResult(summary="", keywords=[], prompt_used="")

        if external_keywords:
            keywords = [Keyword(keyword=k, score=0.0) for k in external_keywords if k.strip()]
        elif use_keywords:
            keywords = [Keyword(k, round(s, 6)) for k, s in self._extractor.extract(clean)]
        else:
            keywords = []
        kw_strings = [k.keyword for k in keywords]

        conditioned = use_keywords and bool(kw_strings)
        if conditioned:
            prompt = build_keyword_prompt(kw_strings, clean, max_sentences=self.max_sentences)
        else:
            prompt = build_unconditioned_prompt(clean, max_sentences=self.max_sentences)

        summary = self.llm_client.generate(
            prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens
        ).strip()
        if not summary:
            raise BackendError("The LLM backend returned an empty summary.")

        return AbstractiveResult(
            summary=summary,
            keywords=keywords,
            prompt_used=prompt,
            metadata={"keyword_conditioned": conditioned},
        )
