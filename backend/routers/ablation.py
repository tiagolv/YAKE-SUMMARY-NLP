"""Ablation router — POST /api/summarize/ablation

Runs all three ablation modes (full / no_keywords / keywords_only) in
parallel using a ThreadPoolExecutor and returns a unified comparison result.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from fastapi import APIRouter

from src.evaluation import evaluate_metrics
from src.keyword_extractor import YakeKeywordExtractor
from src.llm_interface import LLMError, create_llm_client
from src.preprocessor import normalize_text, truncate_text
from src.prompt_builder import (
    build_prompt,
    build_prompt_keywords_only,
    build_prompt_no_keywords,
)

from ..schemas import (
    AblationModeResult,
    AblationRequest,
    AblationResponse,
    AlignmentMetrics,
    CoverageMetrics,
    KeywordItem,
    KeywordsResult,
    MetricsResult,
)
from .deps import get_llm_config, get_prompt_config, get_raw_config, get_yake_config

router = APIRouter(prefix="/api", tags=["ablation"])

ABLATION_MODES = ["full", "no_keywords", "keywords_only"]


def _prompt_for_mode(
    mode: str,
    keywords: list[str],
    truncated_text: str,
    prompt_config: Any,
) -> str:
    if mode == "no_keywords":
        return build_prompt_no_keywords(truncated_text, prompt_config)
    if mode == "keywords_only":
        return build_prompt_keywords_only(keywords, prompt_config)
    return build_prompt(keywords, truncated_text, prompt_config)


def _run_single_mode(
    mode: str,
    keywords: list[str],
    truncated_text: str,
    source_text: str,                     # <-- para compression_ratio
    prompt_config: Any,
    llm_config: Any,
    extractor: YakeKeywordExtractor,
) -> AblationModeResult:
    prompt = _prompt_for_mode(mode, keywords, truncated_text, prompt_config)
    try:
        llm = create_llm_client(llm_config)
        summary = llm.generate(prompt)
    except (LLMError, Exception) as exc:
        return AblationModeResult(mode=mode, summary="", prompt=prompt, error=str(exc))

    metrics_raw = evaluate_metrics(
        summary=summary,
        source_keywords=keywords,
        extractor=extractor,
        source_text=source_text,          # <-- passamos o texto original
    )
    kw_cov = metrics_raw["keyword_coverage"]
    kw_align = metrics_raw["summary_alignment"]

    return AblationModeResult(
        mode=mode,
        summary=summary,
        prompt=prompt,
        metrics=MetricsResult(
            compression_ratio=metrics_raw.get("compression_ratio"),
            keyword_coverage=CoverageMetrics(
                coverage=round(kw_cov["coverage"], 4),
                found=kw_cov["found"],
                missing=kw_cov["missing"],
            ),
            summary_alignment=AlignmentMetrics(
                precision=round(kw_align["precision"], 4),
                recall=round(kw_align["recall"], 4),
                summary_keywords=kw_align.get("summary_keywords", []),
            ),
        ),
    )


def _compute_averages(mode_results: list[AblationModeResult]) -> dict[str, Any]:
    """Compute per-mode metric averages for the comparison table."""
    averages: dict[str, Any] = {}
    for result in mode_results:
        if result.error or result.metrics is None:
            averages[result.mode] = {"error": result.error}
            continue
        averages[result.mode] = {
            "coverage": result.metrics.keyword_coverage.coverage,
            "found_count": len(result.metrics.keyword_coverage.found),
            "missing_count": len(result.metrics.keyword_coverage.missing),
            "alignment_precision": result.metrics.summary_alignment.precision,
            "alignment_recall": result.metrics.summary_alignment.recall,
            "compression_ratio": result.metrics.compression_ratio,
            "summary_words": len(result.summary.split()),
        }
    return averages


@router.post("/summarize/ablation", response_model=AblationResponse)
def run_ablation(req: AblationRequest) -> AblationResponse:
    """Run all three ablation modes in parallel and return a unified comparison."""
    clean_text = normalize_text(req.text)

    # ── Shared setup (computed once) ──────────────────────────────────────────
    yake_config = get_yake_config(req.top_k, req.max_ngram_size)
    extractor = YakeKeywordExtractor(yake_config)

    # FORÇAR YAKE local, ignorando external_keywords
    keywords_scored = extractor.extract(clean_text)
    keywords = [kw for kw, _ in keywords_scored]
    keyword_source = "YAKE! (local)"

    prompt_config = get_prompt_config()
    truncated_text = truncate_text(clean_text, prompt_config.max_text_chars)

    raw_cfg = get_raw_config()
    llm_params = {**raw_cfg["llm"], "temperature": req.temperature}
    from src.llm_interface import LLMConfig
    llm_config = LLMConfig(**llm_params)

    kw_result = KeywordsResult(
        source=keyword_source,
        keywords=[
            KeywordItem(keyword=kw, score=round(score, 6))
            for kw, score in keywords_scored
        ],
    )

    # ── Run 3 modes in parallel ───────────────────────────────────────────────
    mode_results: list[AblationModeResult] = [None] * len(ABLATION_MODES)  # type: ignore

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_idx = {
            pool.submit(
                _run_single_mode,
                mode,
                keywords,
                truncated_text,
                clean_text,               # <-- texto fonte para compression
                prompt_config,
                llm_config,
                extractor,
            ): i
            for i, mode in enumerate(ABLATION_MODES)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                mode_results[idx] = future.result()
            except Exception as exc:
                mode_results[idx] = AblationModeResult(
                    mode=ABLATION_MODES[idx],
                    summary="",
                    prompt="",
                    error=str(exc),
                )

    return AblationResponse(
        keywords=kw_result,
        modes=mode_results,
        averages=_compute_averages(mode_results),
    )