"""Ollama client for local LLM inference."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

import requests

from ..validation import require_int
from ._http import post_json_with_retry
from .base import BackendError


class OllamaClient:
    """Client for a local Ollama server with bounded retry and actionable errors."""

    def __init__(
        self,
        model: str = "mistral",
        host: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 2,
        backoff: float = 1.0,
        post: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        env_host = os.environ.get("OLLAMA_HOST")  # useful on WSL / remote hosts
        host = env_host or host
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = require_int("timeout", timeout)
        self.max_retries = require_int("max_retries", max_retries, minimum=0)
        self.backoff = backoff
        self._post = post
        self._sleep = sleep

    def _hint(self) -> str:
        hint = (
            f"Troubleshooting: (1) start the server with 'ollama serve'; "
            f"(2) check the host ({self.host}) or set OLLAMA_HOST; "
            f"(3) make sure the model exists: 'ollama pull {self.model}'."
        )
        if sys.platform.startswith("linux") and "localhost" in self.host:
            hint += " In WSL, OLLAMA_HOST must point to the Windows host IP."
        return hint

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = post_json_with_retry(
            f"{self.host}/api/generate",
            payload,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
            post=self._post,
            sleep=self._sleep,
            service=f"Ollama (model '{self.model}')",
            troubleshooting=self._hint(),
        )
        text = str(data.get("response", "")).strip()
        if not text:
            raise BackendError(
                f"Ollama model '{self.model}' returned an empty response. {self._hint()}"
            )
        return text

    def is_available(self) -> tuple[bool, str]:
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=3)
            res.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Ollama unreachable at {self.host} [{exc}]. {self._hint()}"
        names = {m.get("name", "").split(":")[0] for m in res.json().get("models", [])}
        if names and self.model.split(":")[0] not in names:
            return False, f"Ollama is up but model '{self.model}' is not pulled. {self._hint()}"
        return True, f"Ollama ({self.model}) ready at {self.host}."
