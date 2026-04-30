"""Aggregation helpers for review-level labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS: list[str] = [
    "review_id",
    "sentiment",
    "concept_subtype",
    "confidence",
    "zone_id",
    "time_key",
]
_OUTPUT_COLUMNS: list[str] = [
    "zone_id",
    "time_key",
    "healthy_review_share",
    "subtype_gap",
    "dominant_subtype",
]
_FULL_HALAL_REQUIRED_COLUMNS: list[str] = [
    "restaurant_id",
    "time_key",
    "zone_id",
    "rating",
    "sentiment",
    "halal_relevance",
    "concept_subtype",
    "confidence",
]

FULL_HALAL_FEATURES = [
    "healthy_food_share",
    "salad_bowls_share",
    "mediterranean_bowls_share",
    "healthy_indian_share",
    "smoothie_juice_share",
    "halal_fast_casual_share",
]


def aggregate_review_labels(
    review_labels: pd.DataFrame,
    topic_distribution: pd.DataFrame | None = None,
    include_sentiment_dist: bool = False,
) -> pd.DataFrame:
    """Aggregate GeminiReviewLabel records into zone-time features.

    Parameters
    ----------
    review_labels:
        DataFrame with columns (review_id, sentiment, concept_subtype, confidence,
        zone_id, time_key).
    topic_distribution:
        Optional DataFrame with zone_id and topic_N_share columns to merge in.
    include_sentiment_dist:
        If True, include frac_positive, frac_neutral, frac_negative columns.

    Returns
    -------
    DataFrame with columns (zone_id, time_key, healthy_review_share, subtype_gap,
    dominant_subtype) plus optional topic and sentiment columns.
    """
    if review_labels.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    # Validate required columns are present
    missing = [c for c in _REQUIRED_COLUMNS if c not in review_labels.columns]
    if missing:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    df = review_labels.copy()

    # Filter to confidence >= 0.7
    df = df[df["confidence"] >= 0.7]
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    def _agg_group(grp: pd.DataFrame) -> pd.Series:
        total = len(grp)
        healthy_share = (grp["sentiment"] == "positive").sum() / max(total, 1)

        # subtype_gap: per-subtype normalized counts
        subtype_counts = grp["concept_subtype"].value_counts()
        subtype_norm = subtype_counts / max(total, 1)
        # scalar summary: std of normalized subtype proportions (gap = variance in coverage)
        subtype_gap = float(subtype_norm.std()) if len(subtype_norm) > 1 else 0.0

        dominant = subtype_counts.idxmax() if len(subtype_counts) > 0 else None

        result_dict = {
            "healthy_review_share": float(healthy_share),
            "subtype_gap": float(subtype_gap),
            "dominant_subtype": dominant,
        }

        if include_sentiment_dist:
            for sent in ("positive", "neutral", "negative"):
                result_dict[f"frac_{sent}"] = (grp["sentiment"] == sent).sum() / max(
                    total, 1
                )

        return pd.Series(result_dict)

    result = df.groupby(["zone_id", "time_key"], as_index=False).apply(
        _agg_group, include_groups=False
    )

    # Merge topic distribution if provided
    if (
        topic_distribution is not None
        and not topic_distribution.empty
        and "zone_id" in topic_distribution.columns
    ):
        result = result.merge(topic_distribution, on="zone_id", how="left")

    return result


def aggregate_nlp_features(
    reviews_df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    gemini_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Returns zone-level NLP features: topic shares, sentiment dist, embedding diversity.

    Parameters
    ----------
    reviews_df:
        Must have zone_id, text columns. Index-aligned with embeddings/cluster_labels.
    embeddings:
        (N, D) embedding array.
    cluster_labels:
        (N,) cluster assignment array.
    gemini_labels:
        DataFrame with review_id, sentiment, concept_subtype, confidence, zone_id, time_key.

    Returns
    -------
    DataFrame with zone-level NLP features.
    """
    from src.nlp.topic_model import topic_distribution_per_zone
    from src.nlp.embeddings import compute_zone_embedding_features

    # Topic distribution
    topic_dist = topic_distribution_per_zone(reviews_df, embeddings, cluster_labels)

    # Embedding features (diversity, PCA)
    emb_features = compute_zone_embedding_features(
        reviews_df, embeddings, cluster_labels
    )

    # Sentiment distribution from gemini labels
    sentiment_agg = aggregate_review_labels(
        gemini_labels,
        topic_distribution=topic_dist,
        include_sentiment_dist=True,
    )

    # Merge embedding features
    if not emb_features.empty and "zone_id" in emb_features.columns:
        # Only keep diversity score from embedding features (topic shares already in topic_dist)
        emb_cols = ["zone_id", "embedding_diversity"]
        emb_subset = emb_features[[c for c in emb_cols if c in emb_features.columns]]
        if not emb_subset.empty:
            sentiment_agg = sentiment_agg.merge(emb_subset, on="zone_id", how="left")

    return sentiment_agg


def aggregate_healthy_review_features(review_labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate full Gemini labels into zone-time healthy food demand features.

    This keeps ``not_related`` reviews as a negative baseline so downstream
    modeling can compare healthy food and non-healthy sentiment within the same
    zone-year panel.
    """
    if review_labels.empty:
        return pd.DataFrame(
            columns=[
                "zone_id",
                "time_key",
                "total_review_count",
                "unique_restaurant_count",
                "halal_related_review_count",
                "explicit_halal_review_count",
                "implicit_halal_review_count",
                "not_related_review_count",
                "halal_related_share",
                "explicit_halal_share",
                "implicit_halal_share",
                "not_related_share",
                "implicit_to_explicit_ratio",
                "overall_positive_rate",
                "overall_negative_rate",
                "halal_positive_rate",
                "halal_negative_rate",
                "non_halal_positive_rate",
                "non_halal_negative_rate",
                "avg_rating",
                "avg_confidence",
                "dominant_subtype",
                "subtype_gap",
            ] + FULL_HALAL_FEATURES
        )

    missing = [
        c for c in _FULL_HALAL_REQUIRED_COLUMNS if c not in review_labels.columns
    ]
    if missing:
        return pd.DataFrame()

    df = review_labels.copy()
    df = df.dropna(subset=["zone_id", "time_key"])
    if df.empty:
        return pd.DataFrame()

    df["time_key"] = pd.to_numeric(df["time_key"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["time_key"])
    if df.empty:
        return pd.DataFrame()

    df["time_key"] = df["time_key"].astype(int)
    df["confidence"] = df["confidence"].fillna(0.0)
    df["sentiment"] = df["sentiment"].fillna("neutral")
    df["halal_relevance"] = df["halal_relevance"].fillna("not_related")
    df["concept_subtype"] = df["concept_subtype"].fillna("other")

    def _agg_group(grp: pd.DataFrame) -> pd.Series:
        total = len(grp)
        halal_grp = grp[grp["halal_relevance"] != "not_related"]
        non_halal_grp = grp[grp["halal_relevance"] == "not_related"]

        explicit_count = int((grp["halal_relevance"] == "explicit_halal").sum())
        implicit_count = int((grp["halal_relevance"] == "implicit_halal").sum())
        not_related_count = int((grp["halal_relevance"] == "not_related").sum())

        subtype_counts = halal_grp["concept_subtype"].value_counts()
        dominant_subtype = (
            subtype_counts.idxmax() if not subtype_counts.empty else "other"
        )
        subtype_gap = (
            float((subtype_counts / len(halal_grp)).std())
            if len(halal_grp) > 0 and len(subtype_counts) > 1
            else 0.0
        )

        HEALTHY_CORE_SUBTYPES = {'salad_bowls', 'mediterranean_bowls', 'healthy_indian', 'vegan_grab_and_go', 'smoothie_juice', 'protein_forward_lunch', 'halal_fast_casual'}
        healthy_grp = grp[grp['concept_subtype'].isin(HEALTHY_CORE_SUBTYPES) | (grp['halal_relevance'] != 'not_related')]

        return pd.Series(
            {
                "total_review_count": int(total),
                "unique_restaurant_count": int(grp["restaurant_id"].nunique()),
                "halal_related_review_count": int(len(halal_grp)),
                "explicit_halal_review_count": explicit_count,
                "implicit_halal_review_count": implicit_count,
                "not_related_review_count": not_related_count,
                "halal_related_share": float(len(halal_grp) / total) if total else 0.0,
                "explicit_halal_share": float(explicit_count / total) if total else 0.0,
                "implicit_halal_share": float(implicit_count / total) if total else 0.0,
                "not_related_share": float(not_related_count / total) if total else 0.0,
                "implicit_to_explicit_ratio": (
                    float(implicit_count / explicit_count)
                    if explicit_count
                    else (float(implicit_count) if implicit_count else 0.0)
                ),
                "overall_positive_rate": float((grp["sentiment"] == "positive").mean()),
                "overall_negative_rate": float((grp["sentiment"] == "negative").mean()),
                "halal_positive_rate": (
                    float((halal_grp["sentiment"] == "positive").mean())
                    if len(halal_grp)
                    else 0.0
                ),
                "halal_negative_rate": (
                    float((halal_grp["sentiment"] == "negative").mean())
                    if len(halal_grp)
                    else 0.0
                ),
                "non_halal_positive_rate": (
                    float((non_halal_grp["sentiment"] == "positive").mean())
                    if len(non_halal_grp)
                    else 0.0
                ),
                "non_halal_negative_rate": (
                    float((non_halal_grp["sentiment"] == "negative").mean())
                    if len(non_halal_grp)
                    else 0.0
                ),
                "avg_rating": (
                    float(grp["rating"].mean()) if grp["rating"].notna().any() else 0.0
                ),
                "avg_confidence": float(grp["confidence"].mean()),
                "dominant_subtype": dominant_subtype,
                "subtype_gap": subtype_gap,
                'healthy_food_share': float(len(healthy_grp) / total) if total else 0.0,
                'salad_bowls_share': float((grp['concept_subtype'] == 'salad_bowls').sum() / total) if total else 0.0,
                'mediterranean_bowls_share': float((grp['concept_subtype'] == 'mediterranean_bowls').sum() / total) if total else 0.0,
                'healthy_indian_share': float((grp['concept_subtype'] == 'healthy_indian').sum() / total) if total else 0.0,
                'smoothie_juice_share': float((grp['concept_subtype'] == 'smoothie_juice').sum() / total) if total else 0.0,
                'halal_fast_casual_share': float((grp['halal_relevance'] != 'not_related').sum() / total) if total else 0.0,
            }
        )

    return (
        df.groupby(["zone_id", "time_key"], as_index=False)
        .apply(_agg_group, include_groups=False)
        .sort_values(["zone_id", "time_key"])
        .reset_index(drop=True)
    )

aggregate_full_halal_review_features = aggregate_healthy_review_features  # backward compat alias

