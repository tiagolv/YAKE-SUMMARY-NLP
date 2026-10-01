"""Client for any OpenAI-compatible API (vLLM, LM Studio, Ollama /v1, LocalAI)."""

from __future__ import annotations

import time
from typing import Any, Callable

import requests

from ..validation import require_int
from ._http import post_json_with_retry
from .base import BackendError


class OpenAICompatibleClient:
    """Client for endpoints conforming to the OpenAI Chat Completions API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str = "none",
        timeout: int = 120,
        max_retries: int = 2,
        backoff: float = 1.0,
        post: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = require_int("timeout", timeout)
        self.max_retries = require_int("max_retries", max_retries, minimum=0)
        self.backoff = backoff
        self._post = post
        self._sleep = sleep

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = post_json_with_retry(
            url,
            payload,
            headers=headers,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
            post=self._post,
            sleep=self._sleep,
            service=f"OpenAI-compatible endpoint (model '{self.model}')",
            troubleshooting=f"Check that the server at {self.base_url} is running and the model name is valid.",
        )
        try:
            text = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"Unexpected response schema from {url}: {str(data)[:200]}") from exc
        if not text:
            raise BackendError(f"Endpoint {url} returned an empty completion.")
        return text

    def is_available(self) -> tuple[bool, str]:
        try:
            res = requests.get(f"{self.base_url}/models", timeout=3)
            res.raise_for_status()
        except requests.RequestException as exc:
            return False, f"API endpoint {self.base_url} unreachable: {exc}"
        return True, f"OpenAI-compatible endpoint ready at {self.base_url}."
