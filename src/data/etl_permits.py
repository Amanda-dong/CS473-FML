"""Placeholder ETL for NYC building permit activity."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="permits",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Permit and renovation activity used in neighborhood change features.",
    columns=("permit_date", "nta_id", "permit_type", "job_count"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
