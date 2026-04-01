"""Placeholder ETL for boundary and crosswalk assets."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="boundaries",
    owner="integration",
    spatial_unit="geometry",
    time_grain="static",
    description="NTA, community district, and micro-zone boundary assets.",
    columns=("zone_id", "zone_type", "geometry_wkt"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
