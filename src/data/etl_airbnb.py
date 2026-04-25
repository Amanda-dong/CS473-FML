"""ETL for Inside Airbnb-derived pressure signals.

Attempts to load from local CSV; downloads from Inside Airbnb if not found.
Returns static covariate (no year column) since snapshots are single-date.
Falls back to synthetic borough-seeded data on any failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.geospatial import lat_lon_to_nta

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

_RAW_CSV = Path("data/raw/insideairbnb_nyc_listings.csv")
_RAW_CSV_GZ = Path("data/raw/insideairbnb_nyc_listings.csv.gz")
_DOWNLOAD_URL = (
    "http://data.insideairbnb.com/united-states/ny/new-york-city/"
    "2024-09-04/data/listings.csv.gz"
)

# Borough-prefix → Airbnb seed values (based on 2024 Inside Airbnb NYC data)
_AIRBNB_PARAMS: dict[str, dict[str, float]] = {
    "mn": {"listing_count": 450.0, "entire_home_ratio": 0.72},
    "bk": {"listing_count": 280.0, "entire_home_ratio": 0.61},
    "qn": {"listing_count": 120.0, "entire_home_ratio": 0.55},
    "bx": {"listing_count": 45.0, "entire_home_ratio": 0.48},
    "si": {"listing_count": 30.0, "entire_home_ratio": 0.43},
}

_ZONE_IDS: list[str] = [
    "bk-tandon",
    "bk-downtownbk",
    "bk-williamsburg",
    "bk-navy-yard",
    "bk-fort-greene",
    "bk-crown-hts",
    "bk-sunset-pk",
    "mn-midtown-e",
    "mn-fidi",
    "mn-columbia",
    "mn-nyu-wash-sq",
    "mn-ues-hosp",
    "mn-chelsea",
    "mn-harlem",
    "mn-lic-adj",
    "qn-lic",
    "qn-astoria",
    "qn-flushing",
    "qn-jackson-hts",
    "qn-forest-hills",
    "qn-jamaica",
    "bx-fordham",
    "bx-mott-haven",
    "bx-co-op-city",
    "si-st-george",
    "BK09",
    "MN17",
    "QN31",
    "BX44",
    "SI22",
]

DATASET_SPEC = DatasetSpec(
    name="airbnb",
    owner="data",
    spatial_unit="nta",
    time_grain="static",
    description="Short-term rental density as a housing-pressure static covariate.",
    columns=("nta_id", "listing_count", "entire_home_ratio"),
)


def _build_synthetic_airbnb() -> pd.DataFrame:
    """Generate plausible static Airbnb snapshot using borough-level seeds."""
    rows: list[dict] = []
    for nta_id in _ZONE_IDS:
        prefix = nta_id[:2].lower()
        params = _AIRBNB_PARAMS.get(prefix, _AIRBNB_PARAMS["mn"])
        seed = abs(hash(nta_id)) % 100_000
        rng = np.random.default_rng(seed)
        noise = rng.uniform(0.70, 1.30)
        rows.append(
            {
                "nta_id": nta_id,
                "listing_count": max(1, round(params["listing_count"] * noise)),
                "entire_home_ratio": round(
                    min(
                        0.99,
                        max(
                            0.01, params["entire_home_ratio"] * rng.uniform(0.85, 1.15)
                        ),
                    ),
                    3,
                ),
                "_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate listings to (nta_id,) producing listing_count and entire_home_ratio."""
    lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lon_col = next(
        (c for c in df.columns if c.lower() in ("longitude", "lng", "lon")), None
    )
    type_col = next((c for c in df.columns if "room_type" in c.lower()), None)

    if lat_col is None or lon_col is None:
        logger.warning("etl_airbnb: lat/lon columns not found — returning placeholder")
        return build_empty_frame(DATASET_SPEC)

    df = df[[c for c in [lat_col, lon_col, type_col] if c]].copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    df["nta_id"] = lat_lon_to_nta(df[lat_col], df[lon_col])

    if type_col:
        df["_is_entire"] = (
            df[type_col].str.lower().str.contains("entire", na=False)
        ).astype(int)
        agg = (
            df.groupby("nta_id")
            .agg(
                listing_count=("nta_id", "count"),
                entire_home_ratio=("_is_entire", "mean"),
            )
            .reset_index()
        )
    else:
        agg = df.groupby("nta_id").size().reset_index(name="listing_count")
        agg["entire_home_ratio"] = 0.0

    return agg[list(DATASET_SPEC.columns)]


def _read_local(limit: int) -> pd.DataFrame | None:
    """Try to read local CSV or CSV.GZ. Returns None if neither exists."""
    for path in (_RAW_CSV, _RAW_CSV_GZ):
        if path.exists():
            logger.info("etl_airbnb: loading from %s", path)
            try:
                return pd.read_csv(path, nrows=limit)
            except Exception as exc:
                logger.warning("etl_airbnb: failed to read %s: %s", path, exc)
    return None


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Load Airbnb listings. Downloads from Inside Airbnb if local file missing.
    Falls back to synthetic borough-seeded data if download also fails.
    """
    df = _read_local(limit)

    if df is None:
        try:
            import requests

            logger.info("etl_airbnb: downloading from %s", _DOWNLOAD_URL)
            resp = requests.get(_DOWNLOAD_URL, timeout=30)
            resp.raise_for_status()
            _RAW_CSV_GZ.parent.mkdir(parents=True, exist_ok=True)
            _RAW_CSV_GZ.write_bytes(resp.content)
            df = pd.read_csv(_RAW_CSV_GZ, nrows=limit)
        except Exception as exc:
            logger.warning(
                "etl_airbnb: download failed (%s) — using synthetic fallback", exc
            )
            return _build_synthetic_airbnb()

    result = _transform(df)
    if result.empty:
        logger.warning(
            "etl_airbnb: transform returned empty — using synthetic fallback"
        )
        return _build_synthetic_airbnb()
    return result
