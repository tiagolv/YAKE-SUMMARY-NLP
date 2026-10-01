"""Shared text utilities: robust sentence/passage splitting and stemmed tokenization."""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem.porter import PorterStemmer

# Abbreviations after which a period does NOT end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "e.g", "i.e", "fig", "figs", "eq", "eqs", "dr", "mr", "mrs", "ms", "prof",
        "vs", "cf", "approx", "no", "sec", "ref", "refs", "vol", "st", "inc",
        "ltd", "ca", "resp", "et al", "al",
    }
)
# 'al' only merges when the next chunk starts in lowercase (see below), so
# "Smith et al. The method..." still splits correctly.
_MERGE_ONLY_IF_LOWER = frozenset({"al", "etc"})

_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_stemmer = PorterStemmer()


def _last_word(chunk: str) -> str:
    m = re.search(r"([A-Za-z.]+)[.!?][\"')\]]*$", chunk)
    return m.group(1).lower() if m else ""


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting abbreviations, initials and decimals.

    Blank lines are always treated as hard boundaries.
    """
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        pieces = [p for p in _BOUNDARY.split(block) if p]
        merged: list[str] = []
        for piece in pieces:
            if merged:
                prev = merged[-1]
                last = _last_word(prev)
                starts_lower = piece[:1].islower()
                is_initial = re.search(r"(?:^|\s)[A-Z]\.$", prev) is not None
                if last in _MERGE_ONLY_IF_LOWER:
                    join = starts_lower
                else:
                    join = last in _ABBREVIATIONS or is_initial or starts_lower
                if join:
                    merged[-1] = f"{prev} {piece}"
                    continue
            merged.append(piece)
        sentences.extend(s.strip() for s in merged if s.strip())
    return sentences


def split_into_passages(text: str) -> list[str]:
    """Split text into paragraphs (blank-line separated), falling back to lines."""
    passages = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not passages:
        passages = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return passages


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def stem_tokens(text: str) -> tuple[str, ...]:
    """Lowercase, tokenize and Porter-stem ``text`` (deterministic)."""
    return tuple(_stem(t.lower()) for t in _TOKEN.findall(text))
