"""FastAPI backend for the YAKE + LLM Local Pipeline.

Usage:
    uvicorn backend.main:app --reload --port 8000
    python backend/main.py --config config.yaml --port 8000
"""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    ablation_router,
    judge_router,
    samples_router,
    status_router,
    summarize_router,
)
from .routers.deps import set_config_path

app = FastAPI(
    title="YAKE + LLM Local Pipeline API",
    description=(
        "REST API for keyword extraction (YAKE!) and keyword-guided "
        "summarization with a local LLM. Supports ablation study and "
        "LLM-as-a-Judge evaluation."
    ),
    version="2.0.0",
)

# Allow the React dev server (port 5173) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status_router)
app.include_router(summarize_router)
app.include_router(ablation_router)
app.include_router(judge_router)
app.include_router(samples_router)


@app.get("/")
def root() -> dict:
    return {
        "name": "YAKE + LLM Local Pipeline API",
        "version": "2.0.0",
        "docs": "/docs",
    }


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="YAKE + LLM FastAPI backend")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    set_config_path(args.config)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()