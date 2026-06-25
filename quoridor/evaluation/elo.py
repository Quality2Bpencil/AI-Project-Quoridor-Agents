"""Small Elo helpers for tournament summaries."""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    return rating_a + k * (score_a - expected_a), rating_b + k * (score_b - expected_b)
