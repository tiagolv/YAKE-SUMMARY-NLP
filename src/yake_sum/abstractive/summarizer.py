"""Abstractive summarizer using YAKE keywords and LLM inference."""

from __future__ import annotations

import yake

from ..backends.base import BaseLLMClient
from ..backends.mock import MockLLMClient
from ..models import AbstractiveResult, Keyword
from .prompt import build_keyword_prompt, build_unconditioned_prompt


class AbstractiveSummarizer:
    """Guided abstractive summarizer that conditions LLM output on YAKE keywords."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        language: str = "en",
        max_ngram_size: int = 3,
        top_k: int = 12,
        deduplication_threshold: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.top_k = top_k
        self.deduplication_threshold = deduplication_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            top=self.top_k,
        )

    def summarize(
        self,
        text: str,
        external_keywords: list[str] | None = None,
        use_keywords: bool = True,
    ) -> AbstractiveResult:
        clean_text = text.strip()
        if not clean_text:
            return AbstractiveResult(
                summary="",
                keywords=[],
                prompt_used="",
            )

        if external_keywords is not None:
            keywords = [Keyword(keyword=k, score=0.0) for k in external_keywords]
            kw_strings = external_keywords
        elif use_keywords:
            kw_tuples = self._extractor.extract_keywords(clean_text)
            keywords = [Keyword(keyword=k, score=round(s, 6)) for k, s in kw_tuples]
            kw_strings = [k.keyword for k in keywords]
        else:
            keywords = []
            kw_strings = []

        if use_keywords and kw_strings:
            prompt = build_keyword_prompt(keywords=kw_strings, text=clean_text)
        else:
            prompt = build_unconditioned_prompt(text=clean_text)

        summary = self.llm_client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return AbstractiveResult(
            summary=summary,
            keywords=keywords,
            prompt_used=prompt,
        )
