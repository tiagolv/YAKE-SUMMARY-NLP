"""Status router — GET /api/status"""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import StatusResponse
from .deps import get_llm_config

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Check whether the configured LLM backend is reachable."""
    from src.llm_interface import check_backend

    config = get_llm_config()
    available, message = check_backend(config)

    return StatusResponse(
        available=available,
        message=message,
        backend=config.backend,
        model=config.ollama_model if config.backend == "ollama" else (config.model_path or ""),
    )