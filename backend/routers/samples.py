"""Samples router — serve the pre-loaded sample documents.

GET /api/samples        → list of available sample names
GET /api/samples/{name} → text content of a sample
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["samples"])

SAMPLES_DIR = Path("data/samples")


class SampleListResponse(BaseModel):
    samples: list[str]


class SampleTextResponse(BaseModel):
    name: str
    text: str


@router.get("/samples", response_model=SampleListResponse)
def list_samples() -> SampleListResponse:
    if not SAMPLES_DIR.exists():
        return SampleListResponse(samples=[])
    names = sorted(p.stem for p in SAMPLES_DIR.glob("*.txt"))
    return SampleListResponse(samples=names)


@router.get("/samples/{name}", response_model=SampleTextResponse)
def get_sample(name: str) -> SampleTextResponse:
    # Sanitize: only allow alphanumeric, underscores and hyphens
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    path = SAMPLES_DIR / f"{safe_name}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{safe_name}' not found.")
    return SampleTextResponse(name=safe_name, text=path.read_text(encoding="utf-8").strip())