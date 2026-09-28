"""Prompt builders for keyword-conditioned abstractive summarization."""

from __future__ import annotations

from typing import Iterable


def build_keyword_prompt(
    keywords: Iterable[str],
    text: str,
    system_instruction: str = "You are a concise scientific and technical summarizer.",
) -> str:
    """Build a prompt conditioning the LLM to cover key concepts extracted by YAKE."""
    kw_str = ", ".join(keywords)
    return (
        f"{system_instruction}\n\n"
        "Given the following keywords extracted from the document, write a concise and faithful summary "
        "that explains the central claims and concepts.\n\n"
        f"Keywords: [{kw_str}]\n\n"
        f"Document:\n{text.strip()}"
    )


def build_unconditioned_prompt(
    text: str,
    system_instruction: str = "You are a concise scientific and technical summarizer.",
) -> str:
    """Build a standard summarization prompt without keywords (for baselines and ablation)."""
    return (
        f"{system_instruction}\n\n"
        "Write a concise and faithful summary of the following document.\n\n"
        f"Document:\n{text.strip()}"
    )
