"""Rent-trajectory feature builders."""

from __future__ import annotations

import pandas as pd

_OUTPUT_COLUMNS: list[str] = ["zone_id", "rent_pressure", "mean_assessed_value"]


def build_rent_trajectory_features(rent_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute rent pressure from PLUTO data.

    Parameters
    ----------
    rent_frame:
        PLUTO-derived DataFrame with columns (year, nta_id, assessed_value, commercial_sqft).

    Returns
    -------
    DataFrame with columns (zone_id, rent_pressure, mean_assessed_value).
    No time_key — PLUTO is cross-sectional. Caller broadcasts across years.
    """
    if rent_frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    df = rent_frame.copy()

    # PLUTO is cross-sectional (current property stock). yearbuilt is construction
    # year, NOT an analysis year.  Aggregate per NTA without time dimension, then
    # broadcast across all years present in the panel later.
    grouped = df.groupby("nta_id", as_index=False).agg(
        mean_assessed_value=("assessed_value", "mean"),
        commercial_sqft_total=("commercial_sqft", "sum"),
    )

    global_min = float(grouped["mean_assessed_value"].min())
    global_max = float(grouped["mean_assessed_value"].max())

    grouped["rent_pressure"] = (
        (grouped["mean_assessed_value"] - global_min) / (global_max - global_min + 1e-9)
    ).clip(0.0, 1.0)

    grouped = grouped.rename(columns={"nta_id": "zone_id"})

    # Return without time_key — caller will broadcast or cross-join as needed
    return grouped[["zone_id", "rent_pressure", "mean_assessed_value"]]
