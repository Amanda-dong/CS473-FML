"""ETL for NYC building permit activity (DOB Permit Issuance)."""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="permits",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Permit and renovation activity used in neighborhood change features.",
    columns=("permit_date", "nta_id", "permit_type", "job_count"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Real ETL
# ---------------------------------------------------------------------------

_DATASET_ID = "ipu4-2q9a"


def fetch(limit: int = 50000) -> pd.DataFrame:
    url = f"https://data.cityofnewyork.us/resource/{_DATASET_ID}.json"
    params = {"$limit": limit}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    date_col = next(
        (
            cols_lower[k]
            for k in cols_lower
            if k in ("issueddate", "issuance_date", "filing_date", "issued_date")
            or ("issued" in k and "date" in k)
        ),
        None,
    )
    board_col = next(
        (
            cols_lower[k]
            for k in cols_lower
            if k in ("nta_id", "communityboard", "community_board", "cb_no")
            or ("community" in k and "board" in k)
        ),
        None,
    )
    type_col = next(
        (
            cols_lower[k]
            for k in cols_lower
            if k
            in (
                "permitsub",
                "permit_sub_type",
                "permit_type",
                "permitsubtype",
                "permit_subtype",
            )
        ),
        None,
    )

    rename: dict[str, str] = {}
    if date_col:
        rename[date_col] = "permit_date"
    if board_col:
        rename[board_col] = "nta_id"
    if type_col:
        rename[type_col] = "permit_type"

    df = df.rename(columns=rename)

    if "permit_date" not in df.columns or "nta_id" not in df.columns:
        logger.warning(
            "etl_permits: required columns not found in API response (available: %s)",
            list(df.columns),
        )
        return build_empty_frame(DATASET_SPEC)

    if "permit_type" not in df.columns:
        df["permit_type"] = "unknown"

    df["permit_date"] = pd.to_datetime(df["permit_date"], errors="coerce")
    df = df.dropna(subset=["permit_date"])
    df["year"] = df["permit_date"].dt.year
    df["job_count"] = 1
    agg = df.groupby(["nta_id", "year", "permit_type"], as_index=False)[
        "job_count"
    ].sum()
    agg["permit_date"] = agg["year"].astype(str)
    return agg[list(DATASET_SPEC.columns)].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Fetch and transform real permit data. Raises on failure."""
    raw = fetch(limit)
    return transform(raw)
