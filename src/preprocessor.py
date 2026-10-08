from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if " " not in truncated:
        return truncated + "..."
    return truncated.rsplit(" ", 1)[0] + "..."


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that protects abbreviations, initials and decimals."""
    from yake_sum.text import split_into_sentences

    return split_into_sentences(text)


def fit_context(
    text: str,
    max_chars: int,
    keywords_scored: list[tuple[str, float]] | None = None,
    strategy: str = "keyword_select",
) -> str:
    """Make ``text`` fit ``max_chars`` for the LLM prompt.

    ``strategy="truncate"`` keeps the legacy behaviour (cut the tail). The default
    ``"keyword_select"`` keeps the most keyword-relevant passages, in original order,
    so information located after the cut-off is no longer silently lost.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if strategy == "truncate" or not keywords_scored:
        return truncate_text(text, max_chars)
    from yake_sum.hybrid.summarizer import select_context

    return "\n\n".join(select_context(text, keywords_scored, max_chars))
