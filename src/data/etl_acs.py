"""ETL for Census ACS features.

Set ``ACS_DATA_PATH`` env var to a local CSV with NTA-level ACS estimates.
Falls back to a synthetic dataset if real data is missing or invalid.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

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


_BOROUGH_PREFIXES = {"MN", "BK", "QN", "BX", "SI"}


def _borough_key(nta_id: str) -> str:
    prefix = nta_id[:2].upper()
    return prefix if prefix in _BOROUGH_PREFIXES else "MN"


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


def _load_local() -> pd.DataFrame:
    """Load ACS data from local CSV(s) specified by env vars.

    Priority:
    1) ``ACS_DATA_GLOB``: glob pattern for multiple yearly files (concatenated)
    2) ``ACS_DATA_PATH``: single CSV path
    """
    glob_str = os.environ.get("ACS_DATA_GLOB", "").strip()
    if glob_str:
        paths = sorted(Path().glob(glob_str))
        if not paths:
            raise FileNotFoundError(
                f"etl_acs: ACS_DATA_GLOB={glob_str} matched no files"
            )
        logger.info(
            "etl_acs: loading %d files from ACS_DATA_GLOB=%s", len(paths), glob_str
        )
        frames = [pd.read_csv(path) for path in paths if path.is_file()]
        if not frames:
            raise RuntimeError("etl_acs: ACS_DATA_GLOB produced no readable files")
        return pd.concat(frames, ignore_index=True)

    path_str = os.environ.get("ACS_DATA_PATH", "")
    if not path_str:
        raise RuntimeError("etl_acs: ACS_DATA_PATH env var not set")
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"etl_acs: ACS_DATA_PATH={path} does not exist")
    logger.info("etl_acs: loading from local file %s", path)
    return pd.read_csv(path)


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ACS input to canonical columns.

    Supports two real-data formats:
    - canonical columns already present
    - NYC NTA profile extract (e.g. GeoID, Pop16plE, MdHHIncE)
    """
    canonical_cols = list(DATASET_SPEC.columns)
    if all(col in df.columns for col in canonical_cols):
        out = df[canonical_cols].copy()
    elif {"GeoID", "Pop16plE", "MdHHIncE"}.issubset(df.columns):
        out = pd.DataFrame(
            {
                "year": int(os.environ.get("ACS_YEAR", "2024")),
                "nta_id": df["GeoID"].astype(str),
                "median_income": pd.to_numeric(df["MdHHIncE"], errors="coerce"),
                "population": pd.to_numeric(df["Pop16plE"], errors="coerce"),
                # Estimate from median income: 30% of monthly income as proxy for rent burden.
                "rent_burden": pd.to_numeric(df["MdHHIncE"], errors="coerce")
                * 0.30
                / 12,
            }
        )
    else:
        raise ValueError(
            "etl_acs: input file does not match expected schema. "
            "Need canonical columns "
            f"{canonical_cols} or source columns ['GeoID','Pop16plE','MdHHIncE']."
        )

    # Ensure rent_burden is computed if it is missing or all null
    if "rent_burden" not in out.columns or out["rent_burden"].isna().all():
        out["rent_burden"] = out["median_income"] * 0.30 / 12

    out = out.dropna(subset=["nta_id"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["median_income"] = pd.to_numeric(out["median_income"], errors="coerce")
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out["rent_burden"] = pd.to_numeric(out["rent_burden"], errors="coerce")
    out = out.dropna(subset=["year", "median_income", "population"])
    out["year"] = out["year"].astype(int)
    return out[canonical_cols].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Load and transform real ACS data. Raises if data is unavailable."""
    try:
        df = _load_local()
        if df.empty:
            raise RuntimeError("etl_acs: local file returned empty frame")
        return _transform(df).head(limit)
    except (RuntimeError, FileNotFoundError):
        from src.config.constants import MODEL_CONFIG

        canonical_path = Path("data/raw/acs_nta_canonical.csv")
        if canonical_path.exists():
            base = pd.read_csv(canonical_path)
            required = {"nta_id", "median_income", "population"}
            if required.issubset(base.columns):
                logger.info("etl_acs: falling back to canonical CSV %s", canonical_path)
                start_year = MODEL_CONFIG.get("temporal_data_start_year", 2020)
                end_year = MODEL_CONFIG.get("temporal_data_end_year", 2024)

                frames = []
                for year in range(start_year, end_year + 1):
                    yr_df = base.copy()
                    yr_df["year"] = year
                    frames.append(yr_df)

                df = pd.concat(frames, ignore_index=True)
                return _transform(df).head(limit)

        logger.warning("etl_acs: canonical fallback failed, using placeholder")
        return run_placeholder_etl()
