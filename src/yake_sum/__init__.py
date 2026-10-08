"""YAKE Summarizer package.

High-level interface for keyword extraction, zero-LLM extractive summarization,
keyword-guided abstractive summarization, and long-document hybrid pipelines.
"""

from .abstractive.summarizer import AbstractiveSummarizer
from .backends.base import (
    BackendConnectionError,
    BackendError,
    BackendTimeoutError,
    BaseLLMClient,
)
from .evaluation import compute_rouge_metrics, keyword_coverage, keyword_prf, source_support
from .extractive.summarizer import ExtractiveSummarizer
from .hybrid.summarizer import HybridSummarizer
from .models import AbstractiveResult, ExtractiveResult, HybridResult, Keyword, SummaryResult
from .summarizer import Summarizer

__version__ = "0.2.0"

__all__ = [
    "Summarizer",
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
    "HybridSummarizer",
    "BaseLLMClient",
    "BackendError",
    "BackendConnectionError",
    "BackendTimeoutError",
    "compute_rouge_metrics",
    "keyword_coverage",
    "keyword_prf",
    "source_support",
    "Keyword",
    "SummaryResult",
    "ExtractiveResult",
    "AbstractiveResult",
    "HybridResult",
]
