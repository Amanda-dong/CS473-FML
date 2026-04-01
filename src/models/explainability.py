"""Helpers that turn scores into recommendation-card explanations."""

from __future__ import annotations

from typing import Mapping


def top_positive_drivers(zone_features: Mapping[str, float]) -> list[str]:
    """Return placeholder explanation strings for the frontend team."""

    drivers = []
    if zone_features.get("quick_lunch_demand", 0.0) > 0:
        drivers.append("Strong lunch-demand proxy")
    if zone_features.get("subtype_gap", 0.0) > 0:
        drivers.append("Subtype-level healthy-food gap")
    if zone_features.get("survival_score", 0.0) > 0:
        drivers.append("Commercial survival outlook is acceptable")
    return drivers or ["Explanation rules not configured yet"]


def top_risks(zone_features: Mapping[str, float]) -> list[str]:
    """Return placeholder risk strings for recommendation cards."""

    risks = []
    if zone_features.get("rent_pressure", 0.0) > 0:
        risks.append("Rent pressure may compress margins")
    if zone_features.get("competition_score", 0.0) > 0:
        risks.append("Existing competitors may reduce whitespace")
    return risks or ["Risk rules not configured yet"]
