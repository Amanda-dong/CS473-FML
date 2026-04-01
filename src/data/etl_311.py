"""Placeholder ETL for NYC 311 complaint signals."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="complaints_311",
    owner="data",
    spatial_unit="community_district",
    time_grain="month",
    description="Quality-of-life and complaint signals for coarse neighborhood context.",
    columns=("month", "community_district", "complaint_type", "count"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
