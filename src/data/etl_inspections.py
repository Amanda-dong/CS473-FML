"""ETL for NYC restaurant inspection results (DOHMH)."""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="inspections",
    owner="data",
    spatial_unit="restaurant",
    time_grain="year",
    description="Restaurant inspection grades, closures, and critical violations.",
    columns=("inspection_date", "restaurant_id", "grade", "critical_flag", "nta_id",
             "cuisine_type", "zipcode"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Real ETL
# ---------------------------------------------------------------------------

_DATASET_ID = "43nn-pn8j"


# Zipcode → NTA mapping built from NYC licenses dataset (which has both fields).
# This covers ~290 NYC zipcodes.  Populated lazily on first use.
_ZIP_TO_NTA: dict[str, str] | None = None


def _get_zip_to_nta() -> dict[str, str]:
    """Lazily build/load the zip→NTA mapping from the licenses API."""
    global _ZIP_TO_NTA
    if _ZIP_TO_NTA is not None:
        return _ZIP_TO_NTA

    try:
        url = f"https://data.cityofnewyork.us/resource/w7w3-xahh.json"
        resp = requests.get(url, params={
            "$limit": 50000,
            "$select": "nta,address_zip",
            "$where": "nta IS NOT NULL",
        }, timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        # Most common NTA per zipcode
        mapping = df.groupby("address_zip")["nta"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        _ZIP_TO_NTA = mapping.to_dict()
        logger.info("Built zip→NTA mapping with %d entries", len(_ZIP_TO_NTA))
    except Exception as e:
        logger.warning("Could not build zip→NTA mapping: %s; using fallback", e)
        _ZIP_TO_NTA = {}

    return _ZIP_TO_NTA


def fetch(limit: int = 50000) -> pd.DataFrame:
    """Fetch restaurant inspection records with zipcode for NTA mapping."""
    url = f"https://data.cityofnewyork.us/resource/{_DATASET_ID}.json"
    params = {
        "$limit": limit,
        "$select": "inspection_date,camis,grade,critical_flag,boro,zipcode,cuisine_description,dba",
        "$where": "inspection_date > '2018-01-01'",
        "$order": "inspection_date DESC",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(columns={
        "camis": "restaurant_id",
        "boro": "_boro",
    })
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")

    # Map zipcode → NTA using the licenses-derived crosswalk
    zip_nta = _get_zip_to_nta()
    if zip_nta and "zipcode" in df.columns:
        df["nta_id"] = df["zipcode"].map(zip_nta)
    else:
        df["nta_id"] = pd.Series(dtype=str)

    # Fallback for unmapped zips: use boro prefix + "01"
    _boro_prefix = {
        "manhattan": "MN", "bronx": "BX", "brooklyn": "BK",
        "queens": "QN", "staten island": "SI",
        "1": "MN", "2": "BX", "3": "BK", "4": "QN", "5": "SI",
    }
    unmapped = df["nta_id"].isna()
    if unmapped.any():
        df.loc[unmapped, "nta_id"] = (
            df.loc[unmapped, "_boro"]
            .fillna("")
            .str.strip()
            .str.lower()
            .map(_boro_prefix)
            .fillna("MN")
            + "01"
        )

    df["grade"] = df["grade"].fillna("N")
    df["critical_flag"] = df["critical_flag"].fillna("Not Applicable")
    df["restaurant_id"] = df["restaurant_id"].fillna("UNKNOWN")
    df["cuisine_type"] = df.get("cuisine_description", pd.Series(dtype=str)).fillna("Unknown")
    if "zipcode" not in df.columns:
        df["zipcode"] = ""
    return df[list(DATASET_SPEC.columns)].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Fetch and transform real inspection data. Raises on failure."""
    raw = fetch(limit)
    return transform(raw)
