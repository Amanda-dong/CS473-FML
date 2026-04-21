"""ETL for NTA boundary and crosswalk assets."""

from __future__ import annotations

import logging

import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="boundaries",
    owner="integration",
    spatial_unit="geometry",
    time_grain="static",
    description="NTA, community district, and micro-zone boundary assets.",
    columns=("zone_id", "zone_type", "geometry_wkt"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------


def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Load real NTA boundary data. Raises — requires NYC GeoJSON download."""
    raise RuntimeError(
        "etl_boundaries: No synthetic data. Download NTA boundaries from "
        "https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/7jgb-yadt "
        "and implement the GeoJSON loader."
    )
