"""Unified Summarizer facade for extractive, abstractive, and hybrid modes."""

from __future__ import annotations

from typing import Any

from .abstractive.summarizer import AbstractiveSummarizer
from .backends import BaseLLMClient, MockLLMClient, get_llm_client
from .evaluation.alignment import keyword_coverage, source_support
from .evaluation.rouge import compute_rouge_metrics
from .extractive.summarizer import ExtractiveSummarizer
from .hybrid.summarizer import HybridSummarizer
from .models import SummaryResult


class Summarizer:
    """Unified summarization interface supporting extractive, abstractive, and hybrid modes."""

    def __init__(
        self,
        mode: str = "extractive",
        backend: str = "mock",
        model: str = "mistral",
        num_sentences: int = 3,
        top_k: int = 15,
        max_ngram_size: int = 3,
        language: str = "en",
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_context_chars: int = 3000,
        llm_client: BaseLLMClient | None = None,
        **kwargs: Any,
    ) -> None:
        self.mode = mode.lower().strip()
        if self.mode not in ("extractive", "abstractive", "hybrid"):
            raise ValueError(
                f"Unknown mode '{mode}'. Choose from 'extractive', 'abstractive', 'hybrid'."
            )
        self.language = language
        self.top_k = top_k
        self.max_ngram_size = max_ngram_size
        self.num_sentences = num_sentences
        self.max_context_chars = max_context_chars

        if self.mode in ("abstractive", "hybrid"):
            self.llm_client = llm_client or get_llm_client(
                backend=backend,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            self.llm_client = None

        # Instantiate sub-engines
        if self.mode == "extractive":
            self._engine = ExtractiveSummarizer(
                num_sentences=num_sentences,
                language=language,
                max_ngram_size=max_ngram_size,
                top_k=top_k,
            )
        elif self.mode == "abstractive":
            self._engine = AbstractiveSummarizer(
                llm_client=self.llm_client,
                language=language,
                max_ngram_size=max_ngram_size,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif self.mode == "hybrid":
            self._engine = HybridSummarizer(
                llm_client=self.llm_client,
                max_context_chars=max_context_chars,
                language=language,
                max_ngram_size=max_ngram_size,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose from 'extractive', 'abstractive', 'hybrid'."
            )

    def summarize(
        self,
        text: str,
        reference_summary: str | None = None,
        external_keywords: list[str] | None = None,
    ) -> SummaryResult:
        if self.mode == "extractive":
            result = self._engine.summarize(text)
            summary_text = result.summary
            keywords = result.keywords
            metrics: dict[str, Any] = {
                "compression_ratio": result.compression_ratio,
                "selected_sentences": result.selected_sentences,
            }
        elif self.mode == "abstractive":
            result = self._engine.summarize(text, external_keywords=external_keywords)
            summary_text = result.summary
            keywords = result.keywords
            metrics = {
                "prompt_used": result.prompt_used,
            }
        elif self.mode == "hybrid":
            result = self._engine.summarize(text)
            summary_text = result.summary
            keywords = result.keywords
            metrics = {
                "retained_passages": result.retained_passages,
                "context_char_count": result.context_char_count,
                "prompt_used": result.prompt_used,
            }

        warnings: list[str] = []
        if self.mode in ("abstractive", "hybrid") and isinstance(self.llm_client, MockLLMClient):
            warnings.append(
                "Mock backend in use: the output is a deterministic placeholder, NOT a real "
                "summary. Use backend='ollama' (or another real backend) for actual results."
            )
        if self.mode == "abstractive" and len(text.strip()) > self.max_context_chars:
            warnings.append(
                f"Input has {len(text.strip())} chars (> max_context_chars={self.max_context_chars}); "
                "the whole text was sent to the LLM and may exceed its context window. "
                "Use mode='hybrid' for long documents."
            )
        if self.mode in ("abstractive", "hybrid") and summary_text:
            support = source_support(summary_text, text)
            metrics["faithfulness"] = support
            if support["unsupported_numbers"]:
                warnings.append(
                    "Summary contains numbers not present in the source: "
                    + ", ".join(support["unsupported_numbers"])
                )
        if not keywords and text.strip():
            warnings.append("No keywords could be extracted; summary is not keyword-conditioned.")
        if warnings:
            metrics["warnings"] = warnings

        # Reference-free sanity metric: how many of the document keywords the summary covers.
        if keywords and summary_text:
            metrics["keyword_coverage"] = keyword_coverage(
                summary_text, [k.keyword for k in keywords]
            )

        # Calculate ROUGE if reference summary is provided
        if reference_summary and summary_text:
            metrics["rouge"] = compute_rouge_metrics(
                summary=summary_text, reference=reference_summary
            )

        return SummaryResult(
            text=summary_text,
            mode=self.mode,
            keywords=keywords,
            metrics=metrics,
        )
