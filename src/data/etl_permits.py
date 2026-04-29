"""ETL for NYC building permit activity (DOB Permit Issuance)."""

from __future__ import annotations

import logging
import re

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
_CB_PREFIX = {"1": "MN", "2": "BX", "3": "BK", "4": "QN", "5": "SI"}
_VALID_PREFIXES = {"MN", "BK", "QN", "BX", "SI"}


def _normalize_nta_like(value: object) -> str:
    """Normalize community-district / NTA-like ids into a stable string code."""
    text = str(value).strip().upper()
    if not text:
        return ""

    # Already normalized forms like BK09, QN70, BK0101
    if re.fullmatch(r"[A-Z]{2}\d{2,4}", text):
        return text

    # Numeric community district codes (e.g., 401, 212, 105) -> QN01, BX12, MN05
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 3 and digits[0] in _CB_PREFIX:
        boro = _CB_PREFIX[digits[0]]
        district = int(digits[-2:])
        return f"{boro}{district:02d}"

    return text


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

    df["nta_id"] = df["nta_id"].map(_normalize_nta_like)
    df = df[df["nta_id"].astype(str).str.strip() != ""].copy()
    df = df[df["nta_id"].astype(str).str[:2].isin(_VALID_PREFIXES)].copy()

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
