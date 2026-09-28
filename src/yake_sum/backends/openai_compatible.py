"""Client for any OpenAI-compatible API (vLLM, LM Studio, Ollama /v1, LocalAI)."""

from __future__ import annotations

import requests
from .base import BackendConnectionError, BackendError


class OpenAICompatibleClient:
    """Client for endpoints conforming to the OpenAI Chat Completions API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str = "none",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

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
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise BackendConnectionError(
                f"Failed to connect to OpenAI endpoint at {self.base_url}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise BackendError(f"API request failed: {exc}") from exc

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def is_available(self) -> tuple[bool, str]:
        try:
            res = requests.get(f"{self.base_url}/models", timeout=3)
            res.raise_for_status()
        except requests.RequestException as exc:
            return False, f"API endpoint {self.base_url} unreachable: {exc}"
        return True, f"OpenAI-compatible endpoint ready at {self.base_url}."
