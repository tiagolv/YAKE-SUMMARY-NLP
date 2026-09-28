"""Sentence scoring based on YAKE keyword significance inversion."""

from __future__ import annotations

import math
import re


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class YakeSentenceScorer:
    """Scores sentences using YAKE keyword weights."""

    def __init__(
        self,
        keywords_scored: list[tuple[str, float]],
        length_penalty_alpha: float = 0.8,
        use_position_bonus: bool = True,
    ) -> None:
        self.length_penalty_alpha = length_penalty_alpha
        self.use_position_bonus = use_position_bonus
        # In YAKE, smaller score = more important.
        # Invert score to get weight: W = 1 / (score + 1e-6)
        self.keyword_weights: list[tuple[str, float]] = []
        for kw, score in keywords_scored:
            norm_kw = _normalize(kw)
            if norm_kw:
                weight = 1.0 / (score + 1e-6)
                self.keyword_weights.append((norm_kw, weight))

    def score_sentence(self, sentence: str, index: int) -> float:
        norm_sent = _normalize(sentence)
        if not norm_sent:
            return 0.0

        words = norm_sent.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0

        raw_score = 0.0
        for norm_kw, weight in self.keyword_weights:
            # Word-boundary aware or substring matching for multi-word phrases
            if f" {norm_kw} " in f" {norm_sent} ":
                raw_score += weight

        # Length normalization
        norm_score = raw_score / math.pow(word_count + 1, self.length_penalty_alpha)

        # Position bonus: lead effect
        if self.use_position_bonus:
            pos_bonus = 1.0 + (1.0 / math.sqrt(index + 1))
        else:
            pos_bonus = 1.0

        return norm_score * pos_bonus

    def score_sentences(self, sentences: list[str]) -> list[float]:
        return [self.score_sentence(s, i) for i, s in enumerate(sentences)]
