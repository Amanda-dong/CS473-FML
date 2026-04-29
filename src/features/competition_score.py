"""Competition scoring for healthy-food concepts."""

from __future__ import annotations

from typing import Mapping


def compute_competition_score(zone_features: Mapping[str, float]) -> float:
    """Score competitive pressure in a zone (higher = more competition).

    Weights: direct_competitors 50%, chain_density 30%, subtype_saturation 20%.
    All inputs are expected in [0, 1]; result is clamped to [0, 1].
    """

    direct_competitors = zone_features.get("direct_competitors", 0.0)
    chain_density = zone_features.get("chain_density", 0.0)
    subtype_saturation = zone_features.get("subtype_saturation", 0.0)
    return round(
        (direct_competitors * 0.5) + (chain_density * 0.3) + (subtype_saturation * 0.2),
        3,
    )
