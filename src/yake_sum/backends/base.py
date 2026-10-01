"""Protocol and base exceptions for LLM backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Base exception for LLM backend failures."""


class BackendConnectionError(BackendError):
    """Raised when an LLM service is unreachable (after bounded retries)."""


class BackendTimeoutError(BackendError):
    """Raised when an LLM service does not answer within the configured timeout."""


@runtime_checkable
class BaseLLMClient(Protocol):
    """Protocol that all LLM backends must implement."""

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response text from the given prompt."""
        ...

    def is_available(self) -> tuple[bool, str]:
        """Check if backend is ready to accept requests. Returns (available, message)."""
        ...
