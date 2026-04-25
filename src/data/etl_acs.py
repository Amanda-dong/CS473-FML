"""ETL for Census ACS features.

Set ``ACS_DATA_PATH`` env var to a local CSV with NTA-level ACS 5-year estimates.
Falls back to synthetic borough-seeded data when the env var is unset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="acs",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Demographic and housing context from ACS 5-year estimates.",
    columns=("year", "nta_id", "median_income", "population", "rent_burden"),
)

# Borough-prefix → demographic seed values (based on NYC Planning 2023 estimates)
_BOROUGH_PARAMS: dict[str, dict[str, float]] = {
    "MN": {"median_income": 95000.0, "population": 12000.0, "rent_burden": 0.38},
    "BK": {"median_income": 68000.0, "population": 18000.0, "rent_burden": 0.45},
    "QN": {"median_income": 72000.0, "population": 22000.0, "rent_burden": 0.40},
    "BX": {"median_income": 42000.0, "population": 25000.0, "rent_burden": 0.52},
    "SI": {"median_income": 78000.0, "population": 14000.0, "rent_burden": 0.33},
}

# Micro-zone IDs used by the recommendation engine
_ZONE_IDS: list[str] = [
    "bk-tandon", "bk-downtownbk", "bk-williamsburg", "bk-navy-yard",
    "bk-fort-greene", "bk-crown-hts", "bk-sunset-pk",
    "mn-midtown-e", "mn-fidi", "mn-columbia", "mn-nyu-wash-sq",
    "mn-ues-hosp", "mn-chelsea", "mn-harlem", "mn-lic-adj",
    "qn-lic", "qn-astoria", "qn-flushing", "qn-jackson-hts",
    "qn-forest-hills", "qn-jamaica",
    "bx-fordham", "bx-mott-haven", "bx-co-op-city",
    "si-st-george",
    "BK09", "MN17", "QN31", "BX44", "SI22",
]

_YEARS = list(range(2018, 2025))

_BOROUGH_PREFIX_MAP = {"bk": "BK", "mn": "MN", "qn": "QN", "bx": "BX", "si": "SI"}


def _borough_key(nta_id: str) -> str:
    prefix2 = nta_id[:2].upper()
    if prefix2 in _BOROUGH_PARAMS:
        return prefix2
    prefix_lower = nta_id[:2].lower()
    return _BOROUGH_PREFIX_MAP.get(prefix_lower, "MN")


def _build_synthetic_acs(limit: int) -> pd.DataFrame:
    """Generate plausible NTA-level ACS data using borough-level seeds."""
    rows: list[dict] = []
    for nta_id in _ZONE_IDS:
        borough = _borough_key(nta_id)
        params = _BOROUGH_PARAMS[borough]
        for year in _YEARS:
            seed = abs(hash(nta_id + str(year))) % 100_000
            rng = np.random.default_rng(seed)
            noise = rng.uniform(0.85, 1.15)
            growth = 1.0 + 0.02 * (year - 2018)  # 2% annual income growth
            rows.append({
                "year": year,
                "nta_id": nta_id,
                "median_income": round(params["median_income"] * growth * noise),
                "population": round(params["population"] * noise),
                "rent_burden": round(
                    min(0.85, max(0.10, params["rent_burden"] * rng.uniform(0.90, 1.10))), 3
                ),
                "_synthetic": True,
            })
    df = pd.DataFrame(rows)
    return df.head(limit)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


def _load_local() -> pd.DataFrame:
    """Load ACS data from a local CSV specified by env var."""
    path_str = os.environ.get("ACS_DATA_PATH", "")
    if not path_str:
        raise RuntimeError("etl_acs: ACS_DATA_PATH env var not set")
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"etl_acs: ACS_DATA_PATH={path} does not exist")
    logger.info("etl_acs: loading from local file %s", path)
    return pd.read_csv(path)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Load real ACS data; falls back to borough-seeded synthetic data if unavailable."""
    try:
        df = _load_local()
        if df.empty:
            raise RuntimeError("etl_acs: local file returned empty frame")
        return df.head(limit)
    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning("etl_acs: real data unavailable (%s) — using synthetic fallback", exc)
        return _build_synthetic_acs(limit)
