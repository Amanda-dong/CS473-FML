"""ETL for Citi Bike mobility features (requires S3 trip data download)."""

from __future__ import annotations

import logging

import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="citibike",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Dock and trip activity as a walkability and lunch-demand proxy.",
    columns=("year", "nta_id", "trip_count", "station_count"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------

def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Load real Citi Bike data. Raises — requires S3 trip data download."""
    raise RuntimeError(
        "etl_citibike: No synthetic data. Download trip data from "
        "https://s3.amazonaws.com/tripdata/index.html and implement the loader."
    )
