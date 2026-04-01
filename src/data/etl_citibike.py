"""Placeholder ETL for Citi Bike mobility features."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="citibike",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Dock and trip activity as a walkability and lunch-demand proxy.",
    columns=("year", "nta_id", "trip_count", "station_count"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
