"""Scoring helpers for healthy-food white space."""

from __future__ import annotations

from typing import Mapping


def score_healthy_gap(zone_features: Mapping[str, float]) -> dict[str, float]:
    """Compute a transparent gap score for a single zone using real demand signals."""

    healthy_supply_ratio = zone_features.get("healthy_supply_ratio", 0.0)
    subtype_gap = zone_features.get("subtype_gap", 0.0)
    quick_lunch_demand = zone_features.get("quick_lunch_demand", 0.0)
    healthy_food_share = zone_features.get("healthy_food_share", zone_features.get("halal_related_share", 0.0))

    score = max(
        0.0,
        (quick_lunch_demand * 0.30)
        + (healthy_food_share * 0.35)
        + (subtype_gap * 0.35)
        - (healthy_supply_ratio * 0.25),
    )

    return {
        "healthy_gap_score": round(score, 3),
        "healthy_supply_ratio": healthy_supply_ratio,
        "subtype_gap": subtype_gap,
        "quick_lunch_demand": quick_lunch_demand,
        "healthy_food_share": healthy_food_share,
    }
