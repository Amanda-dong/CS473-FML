"""Sentence-transformer embeddings for review text."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_EMBEDDING_DIM = 384


@dataclass(frozen=True)
class EmbeddingConfig:
    """Runtime knobs for embedding generation on local or GPU-backed workers."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str | None = None
    batch_size: int = 64
    normalize_embeddings: bool = True


def embed_reviews(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    *,
    config: EmbeddingConfig | None = None,
) -> np.ndarray:
    """Embed review texts using sentence-transformers. Returns (N, 384) array."""
    if not texts:
        return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

    effective_config = config or EmbeddingConfig(model_name=model_name)
    prepared_texts = (
        pd.Series(texts)
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )
    valid_indices = [index for index, text in enumerate(prepared_texts) if text]
    valid_texts = [prepared_texts[index] for index in valid_indices]
    if not valid_texts:
        return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

    def _restore_alignment(valid_embeddings: np.ndarray) -> np.ndarray:
        if len(valid_indices) == len(prepared_texts):
            return valid_embeddings.astype(np.float32)
        aligned = np.zeros((len(prepared_texts), _EMBEDDING_DIM), dtype=np.float32)
        aligned[np.array(valid_indices)] = valid_embeddings
        return aligned

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model_kwargs: dict[str, str] = {}
        if effective_config.device:
            model_kwargs["device"] = effective_config.device
        model = SentenceTransformer(effective_config.model_name, **model_kwargs)
        embeddings = model.encode(
            valid_texts,
            batch_size=effective_config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=effective_config.normalize_embeddings,
        )
        return _restore_alignment(embeddings)
    except ImportError:
        pass

    # Fallback: TF-IDF + TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    n_components = min(_EMBEDDING_DIM, len(valid_texts) - 1, 1000)
    if n_components < 1:
        n_components = 1

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform(valid_texts)

    actual_components = min(n_components, tfidf.shape[1] - 1)
    if actual_components < 1:
        actual_components = 1

    svd = TruncatedSVD(n_components=actual_components, random_state=42)
    reduced = svd.fit_transform(tfidf)

    # Pad to _EMBEDDING_DIM if needed
    if reduced.shape[1] < _EMBEDDING_DIM:
        padding = np.zeros((reduced.shape[0], _EMBEDDING_DIM - reduced.shape[1]), dtype=np.float32)
        reduced = np.hstack([reduced, padding])

    return _restore_alignment(reduced.astype(np.float32))


def optimal_k_search(
    embeddings: np.ndarray,
    k_range: range | None = None,
) -> tuple[int, dict[int, float]]:
    """Find optimal k for K-means via silhouette score.

    Returns (best_k, {k: silhouette_score}).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if k_range is None:
        k_range = range(2, min(15, embeddings.shape[0]))

    scores: dict[int, float] = {}
    for k in k_range:
        if k >= embeddings.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings))))

    best_k = max(scores, key=scores.get) if scores else 2
    return best_k, scores


def cluster_stability(
    embeddings: np.ndarray, n_clusters: int, n_runs: int = 10
) -> float:
    """Measure cluster stability via mean Adjusted Rand Index across random seeds."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    all_labels = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
        all_labels.append(km.fit_predict(embeddings))

    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    return float(np.mean(ari_scores)) if ari_scores else 0.0


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int | None = None
) -> tuple[np.ndarray, object]:
    """K-means clustering on embedding space. Returns (labels, model).

    If n_clusters is None, automatically selects optimal k via silhouette score.
    """
    from sklearn.cluster import KMeans

    if n_clusters is None and embeddings.shape[0] > 3:
        n_clusters, _scores = optimal_k_search(embeddings)
    elif n_clusters is None:
        n_clusters = min(2, embeddings.shape[0])

    n_clusters = min(n_clusters, embeddings.shape[0])
    if n_clusters < 1:
        n_clusters = 1

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    return labels, model


def compute_zone_embedding_features(
    reviews_df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Per-zone: topic distribution, mean embedding (PCA-compressed), diversity score."""
    if reviews_df.empty or "zone_id" not in reviews_df.columns:
        return pd.DataFrame()

    from sklearn.decomposition import PCA

    df = reviews_df.copy()
    df = df.reset_index(drop=True)
    df["_cluster"] = cluster_labels[: len(df)]

    n_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) > 0 else 1

    records = []
    for zone_id, grp in df.groupby("zone_id"):
        idx = grp.index.values
        zone_embs = embeddings[idx]

        # Topic distribution
        cluster_counts = grp["_cluster"].value_counts()
        total = len(grp)
        topic_dist = {}
        for c in range(n_clusters):
            topic_dist[f"topic_{c}_share"] = cluster_counts.get(c, 0) / max(total, 1)

        # Diversity: entropy of cluster distribution
        probs = np.array([topic_dist[f"topic_{c}_share"] for c in range(n_clusters)])
        probs = probs[probs > 0]
        diversity = -float(np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0

        # Mean embedding compressed via first 8 PCA components
        if zone_embs.shape[0] > 1:
            n_comp = min(8, zone_embs.shape[0], zone_embs.shape[1])
            pca = PCA(n_components=n_comp)
            compressed = pca.fit_transform(zone_embs).mean(axis=0)
        else:
            compressed = zone_embs[0][:8] if zone_embs.shape[1] >= 8 else zone_embs[0]

        row = {"zone_id": zone_id, "embedding_diversity": diversity}
        row.update(topic_dist)
        for i, v in enumerate(compressed):
            row[f"emb_pca_{i}"] = float(v)

        records.append(row)

    return pd.DataFrame(records)
