"""Summarize router — POST /api/summarize"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.evaluation import evaluate_metrics
from src.keyword_extractor import YakeKeywordExtractor
from src.llm_interface import LLMError, create_llm_client
from src.preprocessor import fit_context, normalize_text

from ..schemas import (
    AlignmentMetrics,
    CoverageMetrics,
    KeywordItem,
    KeywordsResult,
    MetricsResult,
    SummarizeRequest,
    SummarizeResponse,
)
from .deps import (
    build_prompt_for_mode,
    get_llm_config,
    get_prompt_config,
    get_raw_config,
    get_yake_config,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["summarize"])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    """Run the full YAKE + LLM pipeline for a single document."""
    clean_text = normalize_text(req.text)
    if not clean_text:
        raise HTTPException(status_code=422, detail="O texto está vazio após normalização.")

    # ── Keyword extraction (always offline) ──────────────────────────────────
    yake_config = get_yake_config(req.top_k, req.max_ngram_size)
    extractor = YakeKeywordExtractor(yake_config)

    if req.external_keywords:
        keywords = req.external_keywords
        keywords_scored = [(kw, 0.0) for kw in keywords]
        keyword_source = "external (user-provided)"
    else:
        keywords_scored = extractor.extract(clean_text)
        keywords = [kw for kw, _ in keywords_scored]
        keyword_source = "YAKE! (local)"

    prompt_config = get_prompt_config()
    truncated_text = fit_context(
        clean_text, prompt_config.max_text_chars, keywords_scored, prompt_config.context_strategy
    )
    # Without keywords a "Keywords: []" prompt is worse than a plain summary prompt.
    mode = req.ablation_mode
    if not keywords and mode in ("full", ""):
        mode = "no_keywords"
    prompt = build_prompt_for_mode(mode, req.prompt_template, keywords, truncated_text, prompt_config)

    kw_result = KeywordsResult(
        source=keyword_source,
        keywords=[
            KeywordItem(keyword=kw, score=round(score, 6))
            for kw, score in keywords_scored
        ],
    )

    # ── LLM generation ───────────────────────────────────────────────────────
    raw_cfg = get_raw_config()
    llm_params = {**raw_cfg["llm"], "temperature": req.temperature}
    llm_config = get_llm_config().__class__(**llm_params)

    try:
        llm = create_llm_client(llm_config)
        summary = llm.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - UI must degrade gracefully, but never silently
        if not isinstance(exc, LLMError):
            logger.exception("Unexpected error while generating the summary")
        return SummarizeResponse(
            summary=(
                f"⚠️ Backend LLM inacessível: {exc}\n\n"
                "As keywords e o prompt foram gerados localmente. "
                "Inicie o Ollama (ollama serve) ou configure llama_cpp em "
                "config.yaml para gerar o resumo."
            ),
            keywords=kw_result,
            metrics=None,
            prompt=prompt,
            status=f"Backend LLM inacessível: {exc}",
            backend_available=False,
        )

    # ── Evaluation ───────────────────────────────────────────────────────────
    metrics = evaluate_metrics(
        summary=summary,
        source_keywords=keywords,
        extractor=extractor,
        source_text=clean_text,          # <-- adicionado
    )
    kw_cov = metrics["keyword_coverage"]
    kw_align = metrics["summary_alignment"]

    return SummarizeResponse(
        summary=summary,
        keywords=kw_result,
        metrics=MetricsResult(
            compression_ratio=metrics.get("compression_ratio"),
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
        prompt=prompt,
        status="Resumo gerado com sucesso.",
        backend_available=True,
    )