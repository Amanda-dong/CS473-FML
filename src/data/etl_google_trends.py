"""Deprecated placeholder for Google Trends.

The docs intentionally remove Google Trends from the active plan, but the file is
kept as a low-risk placeholder until the repository cleanup is complete.
"""

from .base import DatasetSpec, build_empty_frame

DATASET_SPEC = DatasetSpec(
    name="google_trends",
    owner="deprecated",
    spatial_unit="metro",
    time_grain="week",
    description="Deprecated unofficial trends source retained for historical context.",
    columns=("week", "term", "interest"),
    status="deprecated",
    notes="Do not use in active feature engineering.",
)


def run_placeholder_etl():
    return build_empty_frame(DATASET_SPEC)


def run_etl(limit: int = 0):  # noqa: ARG001
    """Return an empty frame so deprecated Google Trends never breaks ETL orchestration."""

    return run_placeholder_etl()
