"""ETL for NYC 311 complaint signals."""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="complaints_311",
    owner="data",
    spatial_unit="community_district",
    time_grain="month",
    description="Quality-of-life and complaint signals for coarse neighborhood context.",
    columns=("month", "community_district", "complaint_type", "count"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Real ETL
# ---------------------------------------------------------------------------

_DATASET_ID = "erm2-nwe9"

_FOOD_COMPLAINT_TYPES = (
    "Food Establishment",
    "Unsanitary Food Prep",
    "Food Poisoning",
)

_COMPLAINT_WHERE = (
    "complaint_type IN (" + ", ".join(f"'{c}'" for c in _FOOD_COMPLAINT_TYPES) + ")"
)


def fetch(limit: int = 10000) -> pd.DataFrame:
    url = f"https://data.cityofnewyork.us/resource/{_DATASET_ID}.json"
    params = {
        "$limit": limit,
        "$select": "created_date,community_board,complaint_type",
        "$where": _COMPLAINT_WHERE,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(
        columns={
            "created_date": "_created_date",
            "community_board": "community_district",
        }
    )
    df["_created_date"] = pd.to_datetime(df["_created_date"], errors="coerce")
    df = df.dropna(subset=["_created_date"])
    df["month"] = df["_created_date"].dt.strftime("%Y-%m")
    df["community_district"] = df["community_district"].fillna("Unknown")
    df["complaint_type"] = df["complaint_type"].fillna("Unknown")
    agg = (
        df.groupby(["month", "community_district", "complaint_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return agg[list(DATASET_SPEC.columns)].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Fetch and transform real 311 data. Raises on failure."""
    raw = fetch(limit=min(limit, 10000))
    return transform(raw)
