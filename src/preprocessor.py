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
    # Simple rule-based splitter to avoid heavy NLP deps
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c for c in chunks if c]
