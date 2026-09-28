"""Hybrid summarizer for long documents using extractive filtering before LLM synthesis."""

from __future__ import annotations

import re
import yake

from ..abstractive.prompt import build_keyword_prompt
from ..backends.base import BaseLLMClient
from ..backends.mock import MockLLMClient
from ..extractive.scorer import YakeSentenceScorer
from ..models import HybridResult, Keyword


def split_into_passages(text: str) -> list[str]:
    """Split text by double newlines or paragraph blocks."""
    raw_passages = re.split(r"\n\s*\n", text.strip())
    passages = [p.strip() for p in raw_passages if p.strip()]
    if not passages:
        # Fallback to single lines if no double newlines
        passages = [line.strip() for line in text.splitlines() if line.strip()]
    return passages


class HybridSummarizer:
    """Long-document summarizer that extracts key passages with YAKE before LLM synthesis."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        max_context_chars: int = 3000,
        language: str = "en",
        max_ngram_size: int = 3,
        top_k: int = 15,
        deduplication_threshold: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.max_context_chars = max_context_chars
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

    def summarize(self, text: str) -> HybridResult:
        clean_text = text.strip()
        if not clean_text:
            return HybridResult(
                summary="",
                keywords=[],
                retained_passages=[],
                context_char_count=0,
                prompt_used="",
            )

        # 1. Global keyword extraction over the full document
        kw_tuples = self._extractor.extract_keywords(clean_text)
        keywords = [Keyword(keyword=k, score=round(s, 6)) for k, s in kw_tuples]
        kw_strings = [k.keyword for k in keywords]

        # 2. Segment into passages
        passages = split_into_passages(clean_text)

        # If document already fits within context budget, retain all
        if len(clean_text) <= self.max_context_chars or len(passages) <= 1:
            retained_passages = passages
        else:
            # Score each passage using YakeSentenceScorer
            scorer = YakeSentenceScorer(keywords_scored=kw_tuples)
            scores = scorer.score_sentences(passages)

            # Sort descending by score
            ranked_indices = sorted(
                range(len(passages)), key=lambda i: scores[i], reverse=True
            )

            # Accumulate top passages up to max_context_chars
            selected_indices: list[int] = []
            char_count = 0
            for idx in ranked_indices:
                passage_len = len(passages[idx])
                if char_count + passage_len <= self.max_context_chars or not selected_indices:
                    selected_indices.append(idx)
                    char_count += passage_len
                if char_count >= self.max_context_chars:
                    break

            # Re-sort chronologically by original position in document
            selected_indices.sort()
            retained_passages = [passages[i] for i in selected_indices]

        condensed_text = "\n\n".join(retained_passages)

        # 3. Build prompt and generate summary via LLM
        prompt = build_keyword_prompt(keywords=kw_strings, text=condensed_text)
        summary = self.llm_client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return HybridResult(
            summary=summary,
            keywords=keywords,
            retained_passages=retained_passages,
            context_char_count=len(condensed_text),
            prompt_used=prompt,
        )
