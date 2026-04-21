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
