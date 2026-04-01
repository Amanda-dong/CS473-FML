"""Small ranking helpers for concept-specific recommendations."""

from __future__ import annotations

from typing import Iterable


def rank_zones(scored_rows: Iterable[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Sort scored rows by descending opportunity score."""

    return sorted(scored_rows, key=lambda row: float(row.get("opportunity_score", 0.0)), reverse=True)
