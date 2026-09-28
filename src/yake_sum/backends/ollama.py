"""Ollama client for local LLM inference."""

from __future__ import annotations

import os
import sys
import requests

from .base import BackendConnectionError, BackendError


class OllamaClient:
    """Client for local Ollama server."""

    def __init__(
        self,
        model: str = "mistral",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        # Check environment variable override (useful on WSL)
        env_host = os.environ.get("OLLAMA_HOST")
        self.host = (env_host or host).rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise BackendConnectionError(
                f"Cannot connect to Ollama at {self.host}. Is 'ollama serve' running? [{exc}]"
            ) from exc
        except requests.RequestException as exc:
            raise BackendError(f"Ollama generation failed: {exc}") from exc

        data = response.json()
        return data.get("response", "").strip()

    def is_available(self) -> tuple[bool, str]:
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=3)
            res.raise_for_status()
        except requests.RequestException as exc:
            hint = ""
            if sys.platform.startswith("linux") and "localhost" in self.host:
                hint = " In WSL, set OLLAMA_HOST to the Windows host IP."
            return (
                False,
                f"Ollama unreachable at {self.host} — start server (ollama serve).{hint} [{exc}]",
            )
        return True, f"Ollama ({self.model}) ready at {self.host}."
