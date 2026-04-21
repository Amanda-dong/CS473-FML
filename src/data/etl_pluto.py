"""ETL for PLUTO / MapPLUTO land-use features."""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="pluto",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Built-environment, lot-use, and commercial value proxies.",
    columns=("year", "nta_id", "commercial_sqft", "mixed_use_ratio", "assessed_value"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Real ETL
# ---------------------------------------------------------------------------

_DATASET_ID = "64uk-42ks"


def fetch(limit: int = 50000) -> pd.DataFrame:
    """Fetch PLUTO lot-level data with zipcode for NTA mapping."""
    url = f"https://data.cityofnewyork.us/resource/{_DATASET_ID}.json"
    params = {
        "$limit": limit,
        "$select": "yearbuilt,zipcode,borough,lotarea,bldgarea,comarea,retailarea,assesstot",
        "$where": "comarea > '0' OR retailarea > '0'",  # commercial properties only
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Map zipcode → NTA using the inspections module's shared crosswalk
    try:
        from src.data.etl_inspections import _get_zip_to_nta

        zip_nta = _get_zip_to_nta()
    except Exception:
        zip_nta = {}

    if zip_nta and "zipcode" in df.columns:
        df["nta_id"] = df["zipcode"].map(zip_nta)
    else:
        df["nta_id"] = pd.Series(dtype=str)

    # Fallback: boro prefix + "01"
    _boro_prefix = {"MN": "MN", "BK": "BK", "QN": "QN", "BX": "BX", "SI": "SI"}
    unmapped = df["nta_id"].isna()
    if unmapped.any() and "borough" in df.columns:
        df.loc[unmapped, "nta_id"] = (
            df.loc[unmapped, "borough"].map(_boro_prefix).fillna("MN") + "01"
        )

    df = df.rename(
        columns={
            "yearbuilt": "year",
            "comarea": "commercial_sqft",
            "assesstot": "assessed_value",
        }
    )
    for col in ("year", "commercial_sqft", "bldgarea", "assessed_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "bldgarea" in df.columns:
        df["mixed_use_ratio"] = df.apply(
            lambda r: r["commercial_sqft"] / r["bldgarea"]
            if r["bldgarea"] > 0
            else 0.0,
            axis=1,
        )
    else:
        df["mixed_use_ratio"] = 0.0

    df["year"] = df["year"].astype(int)
    # Drop unmapped rows
    df = df[df["nta_id"].notna() & (df["nta_id"] != "")]
    return df[list(DATASET_SPEC.columns)].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Fetch and transform real PLUTO data. Raises on failure."""
    raw = fetch(limit)
    return transform(raw)
