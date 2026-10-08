from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    backend: str
    model_path: str | None = None
    ollama_model: str = "mistral"
    ollama_host: str = "http://localhost:11434"
    max_tokens: int = 256
    temperature: float = 0.2
    n_ctx: int = 4096


def _ollama_host(config: LLMConfig) -> str:
    """OLLAMA_HOST (e.g. WSL -> Windows host) overrides config.yaml, as the API already did."""
    import os

    return os.environ.get("OLLAMA_HOST") or config.ollama_host


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class LlamaCppClient:
    def __init__(self, config: LLMConfig) -> None:
        if not config.model_path:
            raise LLMError("model_path is required for llama_cpp backend.")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LLMError("Missing dependency 'llama-cpp-python'.") from exc

        self._llm = Llama(model_path=config.model_path, n_ctx=config.n_ctx)
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    def generate(self, prompt: str) -> str:
        result = self._llm(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text = result["choices"][0]["text"]
        return text.strip()


class OllamaClient:
    """Legacy facade over :class:`yake_sum.backends.OllamaClient` (retry + clear errors)."""

    def __init__(self, config: LLMConfig) -> None:
        from yake_sum.backends.ollama import OllamaClient as _Client

        self._client = _Client(model=config.ollama_model, host=_ollama_host(config))
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    def generate(self, prompt: str) -> str:
        from yake_sum.backends.base import BackendError

        try:
            return self._client.generate(
                prompt, temperature=self._temperature, max_tokens=self._max_tokens
            )
        except BackendError as exc:
            raise LLMError(str(exc)) from exc


def create_llm_client(config: LLMConfig) -> LLMClient:
    backend = config.backend.lower()
    if backend == "llama_cpp":
        return LlamaCppClient(config)
    if backend == "ollama":
        return OllamaClient(config)
    if backend == "mock":  # deterministic, for demos and CI (never for quality claims)
        from yake_sum.backends.mock import MockLLMClient

        class _Mock:
            def generate(self, prompt: str) -> str:
                return MockLLMClient().generate(prompt)

        return _Mock()
    raise LLMError(f"Unsupported backend: {config.backend}")


def check_backend(config: LLMConfig) -> tuple[bool, str]:
    """Check whether the configured LLM backend is reachable.

    Returns (available, message) and never raises, so callers (e.g. the UI)
    can render a status indicator without crashing.
    """
    backend = config.backend.lower()

    if backend == "ollama":
        from yake_sum.backends.ollama import OllamaClient as _Client

        return _Client(model=config.ollama_model, host=_ollama_host(config)).is_available()

    if backend == "mock":
        return True, "Mock backend (deterministic; testing only)."

    if backend == "llama_cpp":
        if not config.model_path:
            return False, "llama_cpp: 'model_path' não definido em config.yaml."
        from pathlib import Path

        if not Path(config.model_path).exists():
            return False, f"llama_cpp: modelo não encontrado em {config.model_path}."
        return True, f"llama_cpp: modelo pronto em {config.model_path}."

    return False, f"Backend não suportado: {config.backend}"