"""Demand-feature builders from reviews and social signals."""

from __future__ import annotations

import pandas as pd

_OUTPUT_COLUMNS: list[str] = [
    "zone_id",
    "time_key",
    "healthy_review_share",
    "social_buzz",
]


def build_demand_features(
    review_signals: pd.DataFrame,
    social_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge review and social signals into demand features.

    Parameters
    ----------
    review_signals:
        DataFrame with columns including zone_id, time_key, healthy_review_share.
    social_signals:
        DataFrame with columns including zone_id, time_key, social_buzz (optional).
        May also have 'mention_count' which is renamed to 'social_buzz'.

    Returns
    -------
    DataFrame with columns (zone_id, time_key, healthy_review_share, social_buzz).
    """
    if review_signals.empty and social_signals.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    review_frame = review_signals.copy()
    social_frame = social_signals.copy()

    # Normalize column names
    if (
        "mention_count" in social_frame.columns
        and "social_buzz" not in social_frame.columns
    ):
        social_frame = social_frame.rename(columns={"mention_count": "social_buzz"})

    merged = pd.merge(
        review_frame, social_frame, how="outer", on=["zone_id", "time_key"]
    )

    if "healthy_review_share" not in merged.columns:
        merged["healthy_review_share"] = 0.0
    if "social_buzz" not in merged.columns:
        merged["social_buzz"] = 0.0

    merged["healthy_review_share"] = (
        pd.to_numeric(merged["healthy_review_share"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    merged["social_buzz"] = (
        pd.to_numeric(merged["social_buzz"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )

    return merged


def compute_healthy_review_share(
    yelp_df: pd.DataFrame, taxonomy_keywords: list[str]
) -> float:
    """Compute the fraction of reviews that mention any healthy keyword.

    Parameters
    ----------
    yelp_df:
        DataFrame with a 'review_text' column.
    taxonomy_keywords:
        List of keywords to search for (case-insensitive).

    Returns
    -------
    Float in [0, 1].
    """
    if yelp_df.empty or "review_text" not in yelp_df.columns:
        return 0.0

    if not taxonomy_keywords:
        return 0.0

    texts = yelp_df["review_text"].fillna("").str.lower()
    pattern = "|".join(taxonomy_keywords)
    matches = texts.str.contains(pattern, regex=True, na=False)
    return float(matches.sum()) / max(len(texts), 1)
