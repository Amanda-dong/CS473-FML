"""ETL for Census ACS features.

Set ``ACS_DATA_PATH`` env var to a local CSV with NTA-level ACS 5-year estimates.
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


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Synthetic data (Census API requires a key — not available in course env)
# ---------------------------------------------------------------------------


def _load_local() -> pd.DataFrame:
    """Load ACS data from a local CSV specified by env var."""
    path_str = os.environ.get("ACS_DATA_PATH", "")
    if not path_str:
        raise RuntimeError(
            "etl_acs: ACS_DATA_PATH env var required. "
            "Download ACS 5-year estimates by NTA from NYC Population FactFinder "
            "or Census API and set ACS_DATA_PATH to the CSV."
        )
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"etl_acs: ACS_DATA_PATH={path} does not exist")
    logger.info("etl_acs: loading from local file %s", path)
    return pd.read_csv(path)


def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Load real ACS data from local file. Raises if not configured."""
    return _load_local()
