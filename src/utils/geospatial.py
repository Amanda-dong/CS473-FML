"""Lightweight helpers for describing recommendation micro-zones."""

from __future__ import annotations


def describe_microzone(zone_type: str, label: str) -> str:
    """Return a human-readable description for the UI and docs."""

    descriptions = {
        "campus_walkshed": f"{label} 10-minute campus walkshed",
        "lunch_corridor": f"{label} lunch corridor",
        "transit_catchment": f"{label} transit catchment",
        "business_district": f"{label} business district",
    }
    return descriptions.get(zone_type, label)
