"""Placeholder ETL for Census ACS features."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="acs",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Demographic and housing context from ACS 5-year estimates.",
    columns=("year", "nta_id", "median_income", "population", "rent_burden"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
