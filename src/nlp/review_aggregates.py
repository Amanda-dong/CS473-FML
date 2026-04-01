"""Aggregation helpers for review-level labels."""

from __future__ import annotations

import pandas as pd


def aggregate_review_labels(review_labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate labeled reviews into zone-time features."""

    if review_labels.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "healthy_review_share", "subtype_gap"])
    return review_labels.copy()
