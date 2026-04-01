"""Helpers for subtype-level white-space signals."""

from __future__ import annotations


def compute_subtype_gap(demand_score: float, subtype_supply: float) -> float:
    """Return a simple placeholder gap value for a concept subtype."""

    return round(max(0.0, demand_score - subtype_supply), 3)
