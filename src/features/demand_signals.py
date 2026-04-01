"""Placeholder demand-feature builders from reviews and social signals."""

from __future__ import annotations

import pandas as pd


def build_demand_features(review_signals: pd.DataFrame, social_signals: pd.DataFrame) -> pd.DataFrame:
    """Merge demand-oriented feature tables at a common zone-time key."""

    review_frame = review_signals.copy()
    social_frame = social_signals.copy()
    if review_frame.empty and social_frame.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "healthy_review_share", "social_buzz"])
    return pd.merge(review_frame, social_frame, how="outer", on=["zone_id", "time_key"])
