"""Data models for YAKE Summarizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Keyword:
    keyword: str
    score: float


@dataclass
class ExtractiveResult:
    summary: str
    selected_sentences: list[str]
    keywords: list[Keyword]
    sentence_scores: list[float]
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbstractiveResult:
    summary: str
    keywords: list[Keyword]
    prompt_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridResult:
    summary: str
    keywords: list[Keyword]
    retained_passages: list[str]
    context_char_count: int
    prompt_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryResult:
    text: str
    mode: str
    keywords: list[Keyword]
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
