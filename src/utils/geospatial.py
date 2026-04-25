"""Lightweight helpers for describing recommendation micro-zones."""

from __future__ import annotations

import numpy as np
import pandas as pd


def describe_microzone(zone_type: str, label: str) -> str:
    """Return a human-readable description for the UI and docs."""

    descriptions = {
        "campus_walkshed": f"{label} 10-minute campus walkshed",
        "lunch_corridor": f"{label} lunch corridor",
        "transit_catchment": f"{label} transit catchment",
        "business_district": f"{label} business district",
    }
    return descriptions.get(zone_type, label)


def lat_lon_to_nta(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Vectorized borough-bucket assignment for synthetic NTA-style zone IDs.

    Best-effort proxy until real NTA boundary geometry is available for
    point-in-polygon. Inputs must be aligned numeric Series of equal length.
    """
    lat_arr = lat.to_numpy(dtype=float)
    lon_arr = lon.to_numpy(dtype=float)

    conditions = [
        (lat_arr > 40.78) & (lon_arr < -73.82),
        (lat_arr < 40.65) & (lon_arr < -74.1),
        (lon_arr < -73.95) & (lat_arr >= 40.63) & (lat_arr <= 40.73),
        lon_arr < -73.75,
    ]
    borough = np.select(conditions, ["BX", "SI", "BK", "QN"], default="MN")
    bucket = np.abs(lat_arr * 10).astype(int) % 100
    bucket_str = np.char.zfill(bucket.astype(str), 2)
    return pd.Series(np.char.add(borough.astype(str), bucket_str), index=lat.index)
