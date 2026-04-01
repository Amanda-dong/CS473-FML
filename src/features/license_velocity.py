"""License-velocity feature builders."""

from __future__ import annotations

import pandas as pd


def build_license_velocity_features(license_events: pd.DataFrame) -> pd.DataFrame:
    """Return a placeholder frame with one velocity feature per zone-time bucket."""

    if license_events.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "license_velocity"])
    return license_events.assign(license_velocity=0.0)
