"""Placeholder ETL for Inside Airbnb-derived pressure signals."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="airbnb",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Short-term rental density as a housing-pressure feature.",
    columns=("year", "nta_id", "listing_count", "entire_home_ratio"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
