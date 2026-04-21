"""Competition-score placeholders for healthy-food concepts."""

from __future__ import annotations

from typing import Mapping


def compute_competition_score(zone_features: Mapping[str, float]) -> float:
    """Score competition using the most important placeholder factors."""

    direct_competitors = zone_features.get("direct_competitors", 0.0)
    chain_density = zone_features.get("chain_density", 0.0)
    subtype_saturation = zone_features.get("subtype_saturation", 0.0)
    return round(
        (direct_competitors * 0.5) + (chain_density * 0.3) + (subtype_saturation * 0.2),
        3,
    )
