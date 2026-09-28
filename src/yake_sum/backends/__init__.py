"""Backends for YAKE Summarizer LLM generation."""

from __future__ import annotations

from typing import Any

from .base import BackendConnectionError, BackendError, BaseLLMClient
from .llama_cpp import LlamaCppClient
from .mock import MockLLMClient
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient

__all__ = [
    "BaseLLMClient",
    "BackendError",
    "BackendConnectionError",
    "MockLLMClient",
    "OllamaClient",
    "LlamaCppClient",
    "OpenAICompatibleClient",
    "get_llm_client",
]


def get_llm_client(backend: str = "mock", **kwargs: Any) -> BaseLLMClient:
    """Factory to instantiate the appropriate LLM backend client."""
    name = backend.lower().strip()
    if name == "mock":
        return MockLLMClient(prefix=kwargs.get("prefix", "Mock summary:"))
    if name == "ollama":
        return OllamaClient(
            model=kwargs.get("model", "mistral"),
            host=kwargs.get("host", "http://localhost:11434"),
            timeout=kwargs.get("timeout", 120),
        )
    if name in ("llama_cpp", "llamacpp"):
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("'model_path' is required for llama_cpp backend.")
        return LlamaCppClient(
            model_path=model_path,
            n_ctx=kwargs.get("n_ctx", 4096),
        )
    if name in ("openai", "openai_compatible", "vllm", "lmstudio"):
        return OpenAICompatibleClient(
            base_url=kwargs.get("base_url", "http://localhost:8000/v1"),
            model=kwargs.get("model", "default"),
            api_key=kwargs.get("api_key", "none"),
            timeout=kwargs.get("timeout", 120),
        )
    raise ValueError(
        f"Unsupported backend '{backend}'. Supported: 'mock', 'ollama', 'llama_cpp', 'openai_compatible'"
    )
