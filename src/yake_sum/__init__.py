"""YAKE Summarizer package.

High-level interface for keyword extraction, zero-LLM extractive summarization,
keyword-guided abstractive summarization, and long-document hybrid pipelines.
"""

from .abstractive.summarizer import AbstractiveSummarizer
from .backends.base import BaseLLMClient, BackendError, BackendConnectionError
from .evaluation.rouge import compute_rouge_metrics
from .extractive.summarizer import ExtractiveSummarizer
from .hybrid.summarizer import HybridSummarizer
from .models import AbstractiveResult, ExtractiveResult, HybridResult, Keyword, SummaryResult
from .summarizer import Summarizer

__version__ = "0.1.0"

__all__ = [
    "Summarizer",
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
    "HybridSummarizer",
    "BaseLLMClient",
    "BackendError",
    "BackendConnectionError",
    "compute_rouge_metrics",
    "Keyword",
    "SummaryResult",
    "ExtractiveResult",
    "AbstractiveResult",
    "HybridResult",
]
