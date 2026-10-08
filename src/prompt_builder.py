from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class PromptConfig:
    system: str
    instruction: str
    max_text_chars: int
    context_strategy: str = "keyword_select"  # or "truncate" (legacy behaviour)


def build_prompt(keywords: Iterable[str], text: str, config: PromptConfig) -> str:
    keywords_list = ", ".join(keywords)
    return (
        f"{config.system}\n\n"
        f"{config.instruction}\n\n"
        f"Keywords: [{keywords_list}]\n\n"
        f"Document:\n{text}"
    )


def build_prompt_no_keywords(text: str, config: PromptConfig) -> str:
    """Build a prompt without keywords — text only (for ablation)."""
    return (
        f"{config.system}\n\n"
        "Write a concise summary of the following document.\n\n"
        f"Document:\n{text}"
    )


def build_prompt_keywords_only(keywords: Iterable[str], config: PromptConfig) -> str:
    """Build a prompt with keywords only — no document text (for ablation)."""
    keywords_list = ", ".join(keywords)
    return (
        f"{config.system}\n\n"
        "Based only on the following keywords, write a concise summary "
        "that connects and explains the key concepts.\n\n"
        f"Keywords: [{keywords_list}]"
    )
