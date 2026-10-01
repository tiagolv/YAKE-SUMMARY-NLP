"""Prompt builders for keyword-conditioned abstractive summarization."""

from __future__ import annotations

from typing import Iterable

DEFAULT_SYSTEM = "You are a concise scientific and technical summarizer."
_FAITHFUL = (
    "Use only information stated in the document; do not add facts, numbers or claims "
    "that are not in it. Output only the summary text, with no preamble."
)


def _length_clause(max_sentences: int | None) -> str:
    return f" Write at most {max_sentences} sentences." if max_sentences else ""


def build_keyword_prompt(
    keywords: Iterable[str],
    text: str,
    system_instruction: str = DEFAULT_SYSTEM,
    max_sentences: int | None = 3,
) -> str:
    """Prompt conditioning the LLM to cover the key concepts extracted by YAKE."""
    kw_str = ", ".join(keywords)
    return (
        f"{system_instruction}\n\n"
        "Given the following keywords extracted from the document, write a concise and faithful "
        "summary that covers the central claims and concepts."
        f"{_length_clause(max_sentences)} {_FAITHFUL}\n\n"
        f"Keywords: [{kw_str}]\n\n"
        f"Document:\n{text.strip()}"
    )


def build_unconditioned_prompt(
    text: str,
    system_instruction: str = DEFAULT_SYSTEM,
    max_sentences: int | None = 3,
) -> str:
    """Baseline prompt without keywords (ablation); same length/faithfulness rules."""
    return (
        f"{system_instruction}\n\n"
        "Write a concise and faithful summary of the following document."
        f"{_length_clause(max_sentences)} {_FAITHFUL}\n\n"
        f"Document:\n{text.strip()}"
    )
