"""Placeholder ETL for legally operating business licenses."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="licenses",
    owner="data",
    spatial_unit="restaurant",
    time_grain="year",
    description="Official business-license activity for openings, renewals, and closures.",
    columns=("event_date", "restaurant_id", "license_status", "nta_id", "category"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
