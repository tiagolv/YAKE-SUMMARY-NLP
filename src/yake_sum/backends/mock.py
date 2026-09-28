"""Mock LLM client for deterministic testing and CI without GPU/Ollama."""

from __future__ import annotations

import re


class MockLLMClient:
    """Deterministic mock client that synthesizes consistent summaries from prompts."""

    def __init__(self, prefix: str = "Mock summary:") -> None:
        self.prefix = prefix

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        # Extract keywords if present in prompt like 'Keywords: [kw1, kw2]'
        kw_match = re.search(r"Keywords:\s*\[(.*?)\]", prompt, re.DOTALL)
        if kw_match:
            kws = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]
            kw_part = f" Focused on key concepts: {', '.join(kws[:4])}."
        else:
            kw_part = ""

        # Extract document portion if present
        doc_match = re.search(r"Document:\s*\n?(.*)", prompt, re.DOTALL)
        if doc_match:
            doc_text = doc_match.group(1).strip()
            # Pick first sentence of document
            first_sent = doc_text.split(".")[0].strip()
            if first_sent:
                first_sent += "."
        else:
            first_sent = "The document was analyzed thoroughly."

        return f"{self.prefix} {first_sent}{kw_part}".strip()

    def is_available(self) -> tuple[bool, str]:
        return True, "Mock backend ready for deterministic testing."
