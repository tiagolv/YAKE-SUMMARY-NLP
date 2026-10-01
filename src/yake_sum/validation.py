"""Construction-time validation helpers with clear error messages."""

from __future__ import annotations


def require_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"'{name}' must be >= {minimum}, got {value}.")
    return value


def require_float(name: str, value: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number, got {value!r}.")
    if not (low <= value <= high):
        raise ValueError(f"'{name}' must be between {low} and {high}, got {value}.")
    return float(value)
