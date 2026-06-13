"""Pydantic schemas for request and response validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input document text")
    top_k: int = Field(12, ge=3, le=30)
    max_ngram_size: int = Field(3, ge=1, le=5)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    prompt_template: str = Field("zero_shot")
    ablation_mode: str = Field("full")
    external_keywords: list[str] = Field(default_factory=list)


class AblationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(12, ge=3, le=30)
    max_ngram_size: int = Field(3, ge=1, le=4)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    external_keywords: list[str] = Field(default_factory=list)


class JudgeRequest(BaseModel):
    document: str = Field(..., description="Original document text")
    summary: str = Field(..., description="Summary to evaluate")
    keywords: list[str] = Field(default_factory=list)
    judge_model: str = Field("qwen2.5:7b")


# ── Nested response models ────────────────────────────────────────────────────

class KeywordItem(BaseModel):
    keyword: str
    score: float


class KeywordsResult(BaseModel):
    source: str
    keywords: list[KeywordItem]


class CoverageMetrics(BaseModel):
    coverage: float
    found: list[str]
    missing: list[str]


class AlignmentMetrics(BaseModel):
    precision: float
    recall: float
    summary_keywords: list[str]


class MetricsResult(BaseModel):
    compression_ratio: float | None = None
    keyword_coverage: CoverageMetrics
    summary_alignment: AlignmentMetrics


class AblationModeResult(BaseModel):
    mode: str
    summary: str
    prompt: str
    metrics: MetricsResult | None = None
    error: str | None = None


class JudgeDimension(BaseModel):
    score: int = Field(..., ge=1, le=5)
    justification: str


class JudgeResult(BaseModel):
    fidelity: JudgeDimension
    coverage: JudgeDimension
    coherence: JudgeDimension
    keyword_relevance: JudgeDimension
    overall: float
    model_used: str
    raw_response: str


# ── Top-level responses ───────────────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    summary: str
    keywords: KeywordsResult
    metrics: MetricsResult | None = None
    prompt: str
    status: str
    backend_available: bool


class AblationResponse(BaseModel):
    keywords: KeywordsResult
    modes: list[AblationModeResult]
    averages: dict[str, Any]


class StatusResponse(BaseModel):
    available: bool
    message: str
    backend: str
    model: str