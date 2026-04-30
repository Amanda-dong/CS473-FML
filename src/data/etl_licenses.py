"""ETL for NYC legally operating business licenses."""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="licenses",
    owner="data",
    spatial_unit="restaurant",
    time_grain="year",
    description="Official business-license activity for openings, renewals, and closures.",
    columns=(
        "event_date",
        "restaurant_id",
        "business_unique_id",
        "license_status",
        "nta_id",
        "category",
    ),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Real ETL
# ---------------------------------------------------------------------------

_DATASET_ID = "w7w3-xahh"


def fetch(limit: int = 50000) -> pd.DataFrame:
    """Fetch business licenses with real NTA codes from NYC DCA/SODA.

    This dataset covers DCA-licensed businesses (not DOHMH restaurant permits).
    We fetch all license types as a proxy for commercial activity per NTA.
    For actual restaurant tracking, the inspections dataset (43nn-pn8j) is primary.
    """
    url = f"https://data.cityofnewyork.us/resource/{_DATASET_ID}.json"
    params = {
        "$limit": limit,
        "$select": (
            "license_creation_date,business_unique_id,license_status,"
            "address_borough,nta,address_zip,business_category"
        ),
        "$where": "nta IS NOT NULL",
        "$order": "license_creation_date DESC",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(
        columns={
            "license_creation_date": "event_date",
            "nta": "nta_id",
            "business_category": "category",
        }
    )
    if "business_unique_id" not in df.columns:
        df["business_unique_id"] = pd.NA
    if "restaurant_id" not in df.columns:
        df["restaurant_id"] = pd.NA
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    # Drop placeholder / clearly invalid dates (e.g., 1900-12-31 sentinels).
    df = df[df["event_date"].dt.year >= 2000]
    df["nta_id"] = df["nta_id"].fillna("UNKNOWN")
    df["category"] = df["category"].fillna("Restaurant")
    df["business_unique_id"] = df["business_unique_id"].fillna("UNKNOWN")
    df["license_status"] = df["license_status"].fillna("Unknown")
    # Drop rows with no usable NTA
    df = df[df["nta_id"] != "UNKNOWN"]
    return df[list(DATASET_SPEC.columns)].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Fetch and transform real license data. Raises on failure."""
    raw = fetch(limit)
    return transform(raw)
