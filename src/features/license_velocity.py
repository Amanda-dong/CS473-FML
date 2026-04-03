"""License-velocity feature builders."""

from __future__ import annotations

import pandas as pd

_OPEN_STATUSES: frozenset[str] = frozenset({"Active", "Issued"})
_CLOSE_STATUSES: frozenset[str] = frozenset({"Inactive", "Revoked", "Expired"})
_OUTPUT_COLUMNS: list[str] = ["zone_id", "time_key", "license_velocity", "net_opens", "net_closes"]


def build_license_velocity_features(license_events: pd.DataFrame) -> pd.DataFrame:
    """Compute license velocity features from raw license events DataFrame.

    License velocity = net new restaurant licenses per zone per time period.

    Parameters
    ----------
    license_events:
        DataFrame with columns (event_date, restaurant_id, license_status, nta_id, category).

    Returns
    -------
    DataFrame with columns (zone_id, time_key, license_velocity, net_opens, net_closes).
    """
    if license_events.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    df = license_events.copy()

    # DCA licenses dataset covers all business types (not restaurant-specific).
    # We use total license activity as a commercial vitality proxy per zone.
    # For restaurant-specific tracking, use the inspections dataset.

    # Parse event_date to year
    df["year"] = pd.to_datetime(df["event_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year", "nta_id"])
    df["year"] = df["year"].astype(int)

    # Compute net_opens and net_closes per (nta_id, year)
    df["_is_open"] = df["license_status"].isin(_OPEN_STATUSES).astype(int)
    df["_is_close"] = df["license_status"].isin(_CLOSE_STATUSES).astype(int)

    grouped = df.groupby(["nta_id", "year"], as_index=False).agg(
        net_opens=("_is_open", "sum"),
        net_closes=("_is_close", "sum"),
    )

    grouped["license_velocity"] = grouped["net_opens"] - grouped["net_closes"]
    grouped = grouped.rename(columns={"nta_id": "zone_id", "year": "time_key"})

    return grouped[_OUTPUT_COLUMNS]
