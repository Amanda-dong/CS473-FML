"""ETL for Inside Airbnb-derived pressure signals."""

from __future__ import annotations

import logging

import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="airbnb",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Short-term rental density as a housing-pressure feature.",
    columns=("year", "nta_id", "listing_count", "entire_home_ratio"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------


def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Load real Airbnb data. Raises — requires Inside Airbnb CSV download."""
    raise RuntimeError(
        "etl_airbnb: No synthetic data. Download listings from http://insideairbnb.com/get-the-data "
        "for New York City and implement the loader."
    )
