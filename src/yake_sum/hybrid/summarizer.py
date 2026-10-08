"""Hybrid summarizer: YAKE-guided passage filtering, then LLM synthesis."""

from __future__ import annotations

from ..abstractive.prompt import build_keyword_prompt, build_unconditioned_prompt
from ..backends.base import BackendError, BaseLLMClient
from ..backends.mock import MockLLMClient
from ..extractive.scorer import YakeSentenceScorer
from ..keywords import KeywordExtractor
from ..models import HybridResult, Keyword
from ..text import split_into_passages, split_into_sentences
from ..validation import require_float, require_int

__all__ = ["HybridSummarizer", "select_context", "split_into_passages"]

_PARA_SEP = "\n\n"


def _units(passages: list[str], budget: int) -> list[tuple[str, int]]:
    """Return (text, paragraph_id) units; oversized paragraphs are split into sentences."""
    units: list[tuple[str, int]] = []
    for pid, p in enumerate(passages):
        if len(p) <= budget:
            units.append((p, pid))
            continue
        for s in split_into_sentences(p):
            # A single sentence longer than the budget is hard-truncated at a word boundary.
            if len(s) > budget:
                s = s[:budget].rsplit(" ", 1)[0] or s[:budget]
            units.append((s, pid))
    return units


def select_context(
    text: str, kw_tuples: list[tuple[str, float]], max_chars: int
) -> list[str]:
    """Pick the most keyword-relevant passages that fit in ``max_chars`` (original order).

    Texts already within budget are returned whole (split in paragraphs). Oversized
    paragraphs are split into sentences so a single huge paragraph is still filtered
    instead of just truncated. The joined result (``"\\n\\n".join``) never exceeds
    ``max_chars``.
    """
    clean = text.strip()
    passages = split_into_passages(clean)
    if len(clean) <= max_chars:
        return passages

    units = _units(passages, max_chars)
    texts = [u[0] for u in units]
    scorer = YakeSentenceScorer(kw_tuples)
    # Separators cost 2 chars between paragraphs and 1 inside one; budget with 2 (safe).
    idx = scorer.select(texts, max_chars=max_chars, separator_chars=2)
    if not idx:  # nothing fits (cannot happen after truncation, kept as a safeguard)
        idx = [0]

    retained: list[str] = []
    last_pid = None
    for i in idx:  # ``idx`` is sorted => original chronological order preserved
        unit, pid = units[i]
        if retained and pid == last_pid:
            retained[-1] = f"{retained[-1]} {unit}"
        else:
            retained.append(unit)
        last_pid = pid
    return retained


class HybridSummarizer:
    """Long-document summarizer: keeps the most keyword-relevant passages in original order."""

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
        max_sentences: int | None = 3,
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.max_context_chars = require_int("max_context_chars", max_context_chars, minimum=50)
        self.temperature = require_float("temperature", temperature, 0.0, 2.0)
        self.max_tokens = require_int("max_tokens", max_tokens)
        self.max_sentences = None if max_sentences is None else require_int("max_sentences", max_sentences)
        self._extractor = KeywordExtractor(language, max_ngram_size, top_k, deduplication_threshold)

    def _select(self, clean: str, kw_tuples: list[tuple[str, float]]) -> list[str]:
        return select_context(clean, kw_tuples, self.max_context_chars)

    def summarize(self, text: str) -> HybridResult:
        clean = text.strip()
        if not clean:
            return HybridResult("", [], [], 0, "")

        kw_tuples = self._extractor.extract(clean)  # global keywords over the FULL document
        keywords = [Keyword(k, round(s, 6)) for k, s in kw_tuples]
        kw_strings = [k.keyword for k in keywords]

        retained = self._select(clean, kw_tuples)
        condensed = _PARA_SEP.join(retained)

        if kw_strings:
            prompt = build_keyword_prompt(kw_strings, condensed, max_sentences=self.max_sentences)
        else:
            prompt = build_unconditioned_prompt(condensed, max_sentences=self.max_sentences)

        summary = self.llm_client.generate(
            prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens
        ).strip()
        if not summary:
            raise BackendError("The LLM backend returned an empty summary.")

        return HybridResult(
            summary=summary,
            keywords=keywords,
            retained_passages=retained,
            context_char_count=len(condensed),
            prompt_used=prompt,
            metadata={
                "source_chars": len(clean),
                "reduced": len(condensed) < len(clean),
                "keyword_conditioned": bool(kw_strings),
            },
        )
