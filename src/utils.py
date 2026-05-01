"""Shared utilities — math helpers and domain constants for the halal pipeline."""

from __future__ import annotations

import pandas as pd


# Canonical halal-relevant cuisine families — lowercase for CAMIS string matching.
# Both halal_opportunity.py and halal_forecast.py must import from here (not redeclare).
HALAL_CUISINES: frozenset[str] = frozenset(
    {
        "halal",
        "middle eastern",
        "pakistani",
        "bangladeshi",
        "afghan",
        "egyptian",
        "turkish",
        "moroccan",
        "lebanese",
        "persian/iranian",
    }
)


def minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1]. Returns 0.0 if constant or all-null."""
    s = series.astype(float)
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(0.0, index=s.index)
    return (s - min_v) / (max_v - min_v)
