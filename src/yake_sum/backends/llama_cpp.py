"""llama.cpp in-process Python client for local GGUF models."""

from __future__ import annotations

from pathlib import Path
from .base import BackendError


class LlamaCppClient:
    """Client using llama-cpp-python for GGUF model execution."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._llm = None

    def _load_model(self):
        if self._llm is None:
            if not Path(self.model_path).exists():
                raise BackendError(f"Model file not found at {self.model_path}")
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise BackendError(
                    "Missing 'llama-cpp-python'. Install it with: pip install llama-cpp-python"
                ) from exc
            self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx)
        return self._llm

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        llm = self._load_model()
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return result["choices"][0]["text"].strip()

    def is_available(self) -> tuple[bool, str]:
        if not Path(self.model_path).exists():
            return False, f"Model file not found at {self.model_path}."
        return True, f"llama_cpp model ready at {self.model_path}."
