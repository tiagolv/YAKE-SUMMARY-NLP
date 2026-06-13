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
    def __init__(self, config: LLMConfig) -> None:
        self._host = config.ollama_host.rstrip("/")
        self._model = config.ollama_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    def generate(self, prompt: str) -> str:
        url = f"{self._host}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        return data.get("response", "").strip()


def create_llm_client(config: LLMConfig) -> LLMClient:
    backend = config.backend.lower()
    if backend == "llama_cpp":
        return LlamaCppClient(config)
    if backend == "ollama":
        return OllamaClient(config)
    raise LLMError(f"Unsupported backend: {config.backend}")


def check_backend(config: LLMConfig) -> tuple[bool, str]:
    """Check whether the configured LLM backend is reachable.

    Returns (available, message) and never raises, so callers (e.g. the UI)
    can render a status indicator without crashing.
    """
    import sys

    backend = config.backend.lower()

    if backend == "ollama":
        host = config.ollama_host.rstrip("/")
        try:
            response = requests.get(f"{host}/api/tags", timeout=3)
            response.raise_for_status()
        except requests.RequestException as exc:
            hint = ""
            # On Linux/WSL the host may be running Ollama on Windows, where
            # "localhost" does not always resolve to the Windows host.
            if sys.platform.startswith("linux") and "localhost" in host:
                hint = (
                    " Em WSL, 'localhost' pode não alcançar o Ollama no Windows: "
                    "defina OLLAMA_HOST para o IP do host Windows."
                )
            return False, f"Ollama inacessível em {host} — inicie o servidor (ollama serve).{hint} [{exc}]"
        return True, f"Ollama ({config.ollama_model}) acessível em {host}."

    if backend == "llama_cpp":
        if not config.model_path:
            return False, "llama_cpp: 'model_path' não definido em config.yaml."
        from pathlib import Path

        if not Path(config.model_path).exists():
            return False, f"llama_cpp: modelo não encontrado em {config.model_path}."
        return True, f"llama_cpp: modelo pronto em {config.model_path}."

    return False, f"Backend não suportado: {config.backend}"