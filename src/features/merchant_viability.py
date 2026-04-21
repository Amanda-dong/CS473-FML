"""Placeholder merchant-viability feature helpers."""

from __future__ import annotations

from typing import Mapping


def score_merchant_viability(zone_features: Mapping[str, float]) -> dict[str, float]:
    """Return transparent viability components for the recommendation layer."""

    survival = zone_features.get("survival_score", 0.0)
    rent_pressure = zone_features.get("rent_pressure", 0.0)
    competition = zone_features.get("competition_score", 0.0)
    viability = max(0.0, (survival * 0.5) - (rent_pressure * 0.25) - (competition * 0.25))
    return {
        "merchant_viability_score": round(viability, 3),
        "survival_score": survival,
        "rent_pressure": rent_pressure,
        "competition_score": competition,
    }
