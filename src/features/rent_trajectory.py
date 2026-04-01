"""Rent-trajectory feature builders."""

from __future__ import annotations

import pandas as pd


def build_rent_trajectory_features(rent_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a placeholder rent-pressure feature table."""

    if rent_frame.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "rent_pressure"])
    return rent_frame.assign(rent_pressure=0.0)
