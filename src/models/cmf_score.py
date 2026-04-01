"""Interpretable scoring helpers for recommendation MVPs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreComponents:
    """Named pieces of the opening score."""

    healthy_gap_score: float
    subtype_gap_score: float
    merchant_viability_score: float
    competition_penalty: float


def compute_opening_score(components: ScoreComponents) -> float:
    """Compute a transparent weighted score for ranking zones."""

    score = (
        (components.healthy_gap_score * 0.35)
        + (components.subtype_gap_score * 0.25)
        + (components.merchant_viability_score * 0.3)
        - (components.competition_penalty * 0.1)
    )
    return round(score, 3)
