"""Placeholder ETL for Yelp business and review enrichment."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="yelp",
    owner="data",
    spatial_unit="restaurant",
    time_grain="date",
    description="Review text and business metadata for enrichment only after coverage audit.",
    columns=("review_date", "business_id", "restaurant_id", "rating", "review_text"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
