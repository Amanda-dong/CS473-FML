"""Placeholder ETL for Reddit mention signals."""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="reddit",
    owner="nlp",
    spatial_unit="community_district",
    time_grain="month",
    description="Coarse-grained social mention signals for food interest and buzz.",
    columns=("month", "community_district", "mention_text", "subreddit"),
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)
