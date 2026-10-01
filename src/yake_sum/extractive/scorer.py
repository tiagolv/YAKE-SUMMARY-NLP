"""Sentence scoring and selection based on inverted YAKE keyword scores.

YAKE scores are "lower is better", so each keyword gets weight
``W = 1 / (score + 1e-6)`` (normalised by the maximum weight for a stable scale).
Matching is done on Porter-stemmed token sequences, so "recommender systems"
matches "recommender system". Keywords nested inside a longer matched keyword in
the same sentence are not double counted, and selection is greedy with a
redundancy penalty so a summary does not repeat the same concepts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..text import stem_tokens
from ..validation import require_float

_EPS = 1e-6


def _contains(seq: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return False
    return any(seq[i : i + m] == sub for i in range(n - m + 1))


@dataclass(frozen=True)
class _Kw:
    stems: tuple[str, ...]
    weight: float


class YakeSentenceScorer:
    """Scores and selects sentences (or passages) using YAKE keyword weights."""

    def __init__(
        self,
        keywords_scored: list[tuple[str, float]],
        length_penalty_alpha: float = 0.8,
        use_position_bonus: bool = True,
        redundancy_decay: float = 0.0,
        weight_power: float = 1.0,
    ) -> None:
        self.length_penalty_alpha = require_float(
            "length_penalty_alpha", length_penalty_alpha, 0.0, 2.0
        )
        self.use_position_bonus = use_position_bonus
        self.redundancy_decay = require_float("redundancy_decay", redundancy_decay, 0.0, 1.0)
        self.weight_power = require_float("weight_power", weight_power, 0.1, 2.0)

        # Raw inverted weights (kept for introspection / backward compatibility).
        self.keyword_weights: list[tuple[str, float]] = []
        best: dict[tuple[str, ...], float] = {}
        for kw, score in keywords_scored:
            stems = stem_tokens(kw)
            if not stems:
                continue
            weight = 1.0 / (max(float(score), 0.0) + _EPS)
            self.keyword_weights.append((" ".join(stems), weight))
            best[stems] = max(best.get(stems, 0.0), weight)  # duplicate keywords: keep best

        top = max(best.values(), default=0.0)
        self._keywords: list[_Kw] = [
            _Kw(stems, (w / top) ** self.weight_power) for stems, w in best.items()
        ] if top > 0 else []

    @property
    def has_keywords(self) -> bool:
        return bool(self._keywords)

    # ------------------------------------------------------------------ matching
    def matched_keywords(self, sentence: str) -> list[_Kw]:
        """Distinct keywords found in ``sentence``; nested shorter ones are dropped."""
        stems = stem_tokens(sentence)
        hits = [k for k in self._keywords if _contains(stems, k.stems)]
        return [
            k for k in hits
            if not any(o is not k and len(o.stems) > len(k.stems) and _contains(o.stems, k.stems)
                       for o in hits)
        ]

    def _position_bonus(self, index: int) -> float:
        return 1.0 + 1.0 / math.sqrt(index + 1) if self.use_position_bonus else 1.0

    def _length_norm(self, sentence: str) -> float:
        n = len(stem_tokens(sentence))
        return math.pow(n + 1, self.length_penalty_alpha) if n else float("inf")

    def score_sentence(self, sentence: str, index: int) -> float:
        raw = sum(k.weight for k in self.matched_keywords(sentence))
        if raw == 0.0:
            return 0.0
        return raw / self._length_norm(sentence) * self._position_bonus(index)

    def score_sentences(self, sentences: list[str]) -> list[float]:
        return [self.score_sentence(s, i) for i, s in enumerate(sentences)]

    # ----------------------------------------------------------------- selection
    def select(
        self,
        units: list[str],
        max_units: int | None = None,
        max_chars: int | None = None,
        separator_chars: int = 1,
    ) -> list[int]:
        """Greedy selection returning chronologically sorted indices.

        Each round picks the unit with the highest *marginal* gain: keywords that
        are already covered by the selection count only ``redundancy_decay`` of their
        weight. Ties (and the no-keyword case) fall back to document order, i.e. Lead-k.
        Stops at ``max_units`` and/or when nothing else fits within ``max_chars``.
        """
        if not units:
            return []
        matches = [self.matched_keywords(u) for u in units]
        norms = [self._length_norm(u) for u in units]
        bonus = [self._position_bonus(i) for i in range(len(units))]
        covered: set[tuple[str, ...]] = set()
        chosen: list[int] = []
        used = 0
        limit = max_units if max_units is not None else len(units)

        while len(chosen) < limit:
            best_i, best_gain = -1, -1.0
            for i, u in enumerate(units):
                if i in chosen:
                    continue
                cost = len(u) + (separator_chars if chosen else 0)
                if max_chars is not None and used + cost > max_chars:
                    continue
                raw = sum(
                    k.weight * (self.redundancy_decay if k.stems in covered else 1.0)
                    for k in matches[i]
                )
                gain = raw / norms[i] * bonus[i] + 1e-9 * bonus[i]  # tie-break: earlier first
                if gain > best_gain:
                    best_i, best_gain = i, gain
            if best_i < 0:
                break
            chosen.append(best_i)
            used += len(units[best_i]) + (separator_chars if len(chosen) > 1 else 0)
            covered.update(k.stems for k in matches[best_i])
        return sorted(chosen)
