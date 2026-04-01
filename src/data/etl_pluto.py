"""Placeholder ETL for PLUTO and MapPLUTO features."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="pluto",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Built-environment, lot-use, and commercial value proxies.",
    columns=("year", "nta_id", "commercial_sqft", "mixed_use_ratio", "assessed_value"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
