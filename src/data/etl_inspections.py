"""Placeholder ETL for NYC restaurant inspection data."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="inspections",
    owner="data",
    spatial_unit="restaurant",
    time_grain="year",
    description="Restaurant inspection grades, closures, and critical violations.",
    columns=("inspection_date", "restaurant_id", "grade", "critical_flag", "nta_id"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
