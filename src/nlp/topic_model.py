"""Topic modeling helpers for review exploration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def starter_topic_labels() -> tuple[str, ...]:
    """Return initial topic names for exploratory analysis."""

    return ("healthy", "speed", "taste", "price", "service")


def discover_topics(
    embeddings: np.ndarray,
    n_topics: int = 8,
    texts: list[str] | None = None,
) -> dict:
    """Cluster embeddings and extract representative terms per topic.

    Parameters
    ----------
    embeddings:
        (N, D) array of review embeddings.
    n_topics:
        Number of topics/clusters.
    texts:
        Optional list of review texts for term extraction via TF-IDF.

    Returns
    -------
    Dict with keys: cluster_labels (ndarray), topic_terms (dict[int, list[str]]),
    centroids (ndarray).
    """
    from sklearn.cluster import KMeans

    n_topics = min(n_topics, embeddings.shape[0])
    if n_topics < 1:
        n_topics = 1

    km = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    topic_terms: dict[int, list[str]] = {}

    if texts is not None and len(texts) > 0:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        for cluster_id in range(n_topics):
            mask = labels == cluster_id
            if not mask.any():
                topic_terms[cluster_id] = []
                continue
            cluster_tfidf = tfidf[mask].mean(axis=0)
            cluster_arr = np.asarray(cluster_tfidf).flatten()
            top_indices = cluster_arr.argsort()[-10:][::-1]
            topic_terms[cluster_id] = [feature_names[i] for i in top_indices]
    else:
        for cluster_id in range(n_topics):
            topic_terms[cluster_id] = []

    return {
        "cluster_labels": labels,
        "topic_terms": topic_terms,
        "centroids": km.cluster_centers_,
    }


def topic_distribution_per_zone(
    reviews_df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute topic share distribution per zone.

    Parameters
    ----------
    reviews_df:
        Must have 'zone_id' column. Index-aligned with embeddings/labels.
    embeddings:
        (N, D) embedding array (unused directly, kept for API consistency).
    cluster_labels:
        (N,) array of cluster assignments.

    Returns
    -------
    DataFrame with zone_id and topic_N_share columns.
    """
    if reviews_df.empty or "zone_id" not in reviews_df.columns:
        return pd.DataFrame()

    df = reviews_df[["zone_id"]].copy().reset_index(drop=True)
    df["_cluster"] = cluster_labels[: len(df)]

    n_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) > 0 else 1

    records = []
    for zone_id, grp in df.groupby("zone_id"):
        total = len(grp)
        counts = grp["_cluster"].value_counts()
        row = {"zone_id": zone_id}
        for c in range(n_clusters):
            row[f"topic_{c}_share"] = counts.get(c, 0) / max(total, 1)
        records.append(row)

    return pd.DataFrame(records)
