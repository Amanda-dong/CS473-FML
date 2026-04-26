"""Tests for NLP helper modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.data.quality import prepare_embedding_corpus
from src.nlp.embeddings import EmbeddingConfig, embed_reviews
from src.nlp.subtype_classifier import classify_subtype
from src.nlp.white_space import compute_subtype_gap


# ── subtype classifier ────────────────────────────────────────────────────────


def test_subtype_classifier_recognizes_indian_gap() -> None:
    assert (
        classify_subtype("Looking for a healthy Indian lunch bowl option")
        == "healthy_indian"
    )


def test_subtype_classifier_recognizes_ramen() -> None:
    assert classify_subtype("Best tonkotsu ramen spot in Brooklyn") == "ramen"


def test_subtype_classifier_recognizes_korean() -> None:
    assert classify_subtype("Authentic Korean BBQ with great bulgogi") == "korean"


def test_subtype_classifier_recognizes_mexican() -> None:
    assert classify_subtype("Freshest tacos al pastor in Queens") == "mexican"


def test_subtype_classifier_unknown_falls_through() -> None:
    result = classify_subtype("Obscure cuisine not in taxonomy xyz")
    assert isinstance(result, str)


def test_batch_classify_returns_correct_length() -> None:
    from src.nlp.subtype_classifier import batch_classify

    texts = ["healthy Indian bowl", "vegan wrap", "ramen shop"]
    results = batch_classify(texts)
    assert len(results) == 3
    assert isinstance(results[0], str)


def test_classify_subtype_embedding_no_centroids() -> None:
    from src.nlp.subtype_classifier import classify_subtype_embedding

    assert (
        classify_subtype_embedding("text", {}, lambda x: np.array([[1.0]])) == "unknown"
    )


def test_classify_subtype_embedding_zero_norm() -> None:
    from src.nlp.subtype_classifier import classify_subtype_embedding

    centroids = {"ramen": np.array([0.0, 0.0])}
    assert (
        classify_subtype_embedding("text", centroids, lambda x: np.array([[0.0, 0.0]]))
        == "unknown"
    )


def test_classify_subtype_embedding_matches_best() -> None:
    from src.nlp.subtype_classifier import classify_subtype_embedding

    centroids = {
        "ramen": np.array([1.0, 0.0]),
        "mexican": np.array([0.0, 1.0]),
        "zero": np.array([0.0, 0.0]),
    }

    def mock_embed(texts):
        return np.array([[0.9, 0.1]])

    assert classify_subtype_embedding("text", centroids, mock_embed) == "ramen"


# ── white space ───────────────────────────────────────────────────────────────


def test_subtype_gap_is_non_negative() -> None:
    assert compute_subtype_gap(0.2, 0.5) == 0.0


def test_subtype_gap_positive_when_demand_exceeds_supply() -> None:
    gap = compute_subtype_gap(0.8, 0.3)
    assert gap == pytest.approx(0.5, abs=0.001)


def test_subtype_gap_exact_balance() -> None:
    assert compute_subtype_gap(0.5, 0.5) == 0.0


# ── review aggregation ────────────────────────────────────────────────────────


def test_review_aggregates_returns_expected_columns(
    sample_review_labels: pd.DataFrame,
) -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    result = aggregate_review_labels(sample_review_labels)
    assert "healthy_review_share" in result.columns
    assert "zone_id" in result.columns
    assert "time_key" in result.columns


def test_review_aggregates_empty_input() -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    result = aggregate_review_labels(pd.DataFrame())
    assert list(result.columns) == [
        "zone_id",
        "time_key",
        "healthy_review_share",
        "subtype_gap",
        "dominant_subtype",
    ]


def test_review_aggregates_share_bounded(sample_review_labels: pd.DataFrame) -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    result = aggregate_review_labels(sample_review_labels)
    if not result.empty:
        assert result["healthy_review_share"].between(0.0, 1.0).all()


# ── gemini labels ─────────────────────────────────────────────────────────────


def test_gemini_label_prompt_contains_subtype() -> None:
    from src.nlp.gemini_labels import build_label_prompt

    prompt = build_label_prompt("great salad", ("salad_bowls", "healthy_indian"))
    assert "salad_bowls" in prompt


def test_gemini_label_prompt_contains_review() -> None:
    from src.nlp.gemini_labels import build_label_prompt

    review = "Amazing Indian lunch"
    prompt = build_label_prompt(review, ("healthy_indian",))
    assert review in prompt


def test_label_reviews_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify labeling raises when no API key is provided."""
    from src.nlp.gemini_labels import label_reviews_with_gemini

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reviews = ["good food", "bad service", "amazing tacos"]
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        label_reviews_with_gemini(reviews, ("salad_bowls", "mexican"), api_key=None)


def test_gemini_cache_keys_by_review_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.nlp import gemini_labels

    class FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(
            self, *, model: str, contents: str, config: dict
        ) -> object:  # noqa: ARG002
            self.calls += 1
            if "fresh salad" in contents:
                payload = [
                    {
                        "sentiment": "positive",
                        "concept_subtype": "salad_bowls",
                        "confidence": 0.9,
                        "rationale": "fresh",
                    }
                ]
            else:
                payload = [
                    {
                        "sentiment": "negative",
                        "concept_subtype": "salad_bowls",
                        "confidence": 0.9,
                        "rationale": "greasy",
                    }
                ]
            return type("Resp", (), {"text": json.dumps(payload)})()

    fake_models = FakeModels()

    class FakeClient:
        def __init__(self, api_key: str) -> None:  # noqa: ARG002
            self.models = fake_models

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setattr(
        gemini_labels, "_CACHE_PATH", tmp_path / "gemini_labels.parquet"
    )

    first = gemini_labels.label_reviews_with_gemini(
        ["fresh salad"], ("salad_bowls",), api_key="test-key"
    )
    second = gemini_labels.label_reviews_with_gemini(
        ["greasy burger"], ("salad_bowls",), api_key="test-key"
    )

    assert first[0].sentiment == "positive"
    assert second[0].sentiment == "negative"
    assert fake_models.calls == 2


# ── neighborhood mentions ─────────────────────────────────────────────────────


def test_extract_location_mentions_finds_match() -> None:
    from src.nlp.neighborhood_mentions import extract_location_mentions

    matches = extract_location_mentions(
        "Great food near NYU Tandon campus", ("NYU Tandon", "Midtown")
    )
    assert "NYU Tandon" in matches


def test_extract_location_mentions_empty() -> None:
    from src.nlp.neighborhood_mentions import extract_location_mentions

    matches = extract_location_mentions("No location here", ("Brooklyn", "Astoria"))
    assert matches == []


def test_prepare_embedding_corpus_drops_blank_and_duplicate_reviews() -> None:
    frame = pd.DataFrame(
        {
            "review_text": [
                " great salad bowl  ",
                "",
                "great salad bowl",
                "fresh wraps nearby",
            ],
            "restaurant_id": ["r1", "r1", "r1", "r2"],
        }
    )

    cleaned, report = prepare_embedding_corpus(frame)

    assert cleaned["review_text"].tolist() == ["great salad bowl", "fresh wraps nearby"]
    assert report.dropped_rows == 2


def test_embed_reviews_returns_empty_matrix_for_blank_inputs() -> None:
    embeddings = embed_reviews(
        ["   ", "\n"],
        config=EmbeddingConfig(batch_size=8, device="cuda"),
    )
    assert embeddings.shape == (0, 384)


# ── aggregate_review_labels schema tests ──────────────────────────────────────


def test_aggregate_review_labels_subtype_gap_non_negative(
    sample_review_labels: pd.DataFrame,
) -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    result = aggregate_review_labels(sample_review_labels)
    if not result.empty:
        assert (result["subtype_gap"] >= 0.0).all()


def test_aggregate_review_labels_dominant_subtype_is_string(
    sample_review_labels: pd.DataFrame,
) -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    result = aggregate_review_labels(sample_review_labels)
    if not result.empty and "dominant_subtype" in result.columns:
        non_null = result["dominant_subtype"].dropna()
        assert non_null.apply(lambda v: isinstance(v, str)).all()


def test_build_zone_year_matrix_accepts_311_for_social_buzz() -> None:
    from src.features.feature_matrix import build_zone_year_matrix

    complaints = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03"],
            "community_district": ["Brooklyn", "Manhattan", "Harlem"],
            "complaint_type": ["Food Establishment"] * 3,
            "count": [5, 8, 12],
        }
    )
    result = build_zone_year_matrix({"complaints_311": complaints})
    assert isinstance(result, pd.DataFrame)


# ── embeddings (additional) ────────────────────────────────────────────────────


def test_embed_reviews_empty_list() -> None:
    embeddings = embed_reviews([])
    assert embeddings.shape == (0, 384)


def test_embed_reviews_returns_correct_shape() -> None:
    embeddings = embed_reviews(["fresh healthy lunch bowl"])
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 384


def test_embed_reviews_multiple_texts() -> None:
    texts = ["great salad", "ramen noodles", "healthy wrap", "veggie bowl", "tacos"]
    embeddings = embed_reviews(texts)
    assert embeddings.shape == (5, 384)
    assert embeddings.dtype == np.float32


def test_embed_reviews_partial_blanks_alignment() -> None:
    texts = ["hello world", "   ", "another text"]
    embeddings = embed_reviews(texts)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.dtype == np.float32
    # blank row should be zeros
    assert (embeddings[1] == 0.0).all()


def test_embed_reviews_with_custom_config() -> None:
    from src.nlp.embeddings import EmbeddingConfig

    cfg = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2", batch_size=4, normalize_embeddings=False
    )
    embeddings = embed_reviews(["test text one", "test text two"], config=cfg)
    assert embeddings.shape == (2, 384)


def test_embed_reviews_with_device_config() -> None:
    from src.nlp.embeddings import EmbeddingConfig

    # device setting should not crash even if unavailable
    cfg = EmbeddingConfig(device="cpu")
    embeddings = embed_reviews(["healthy indian lunch"], config=cfg)
    assert embeddings.shape == (1, 384)


def test_optimal_k_search_returns_best_k() -> None:
    from src.nlp.embeddings import optimal_k_search

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, 8)).astype(np.float32)
    best_k, scores = optimal_k_search(embeddings, k_range=range(2, 5))
    assert 2 <= best_k <= 4
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_optimal_k_search_default_range() -> None:
    from src.nlp.embeddings import optimal_k_search

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((10, 8)).astype(np.float32)
    best_k, scores = optimal_k_search(embeddings)
    assert best_k >= 2


def test_cluster_stability() -> None:
    from src.nlp.embeddings import cluster_stability

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((15, 8)).astype(np.float32)
    ari = cluster_stability(embeddings, n_clusters=2, n_runs=3)
    assert 0.0 <= ari <= 1.0


def test_cluster_embeddings_fixed_k() -> None:
    from src.nlp.embeddings import cluster_embeddings

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((10, 8)).astype(np.float32)
    labels, model = cluster_embeddings(embeddings, n_clusters=3)
    assert len(labels) == 10
    assert len(set(labels)) <= 3


def test_cluster_embeddings_auto_k_small() -> None:
    from src.nlp.embeddings import cluster_embeddings

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((3, 8)).astype(np.float32)
    labels, _ = cluster_embeddings(embeddings, n_clusters=None)
    assert len(labels) == 3


def test_cluster_embeddings_auto_k_large() -> None:
    from src.nlp.embeddings import cluster_embeddings

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, 8)).astype(np.float32)
    labels, _ = cluster_embeddings(embeddings, n_clusters=None)
    assert len(labels) == 20


def test_compute_zone_embedding_features_empty() -> None:
    from src.nlp.embeddings import compute_zone_embedding_features

    result = compute_zone_embedding_features(
        pd.DataFrame(), np.zeros((0, 8)), np.array([])
    )
    assert result.empty


def test_compute_zone_embedding_features_no_zone_id() -> None:
    from src.nlp.embeddings import compute_zone_embedding_features

    df = pd.DataFrame({"text": ["hello"]})
    result = compute_zone_embedding_features(df, np.zeros((1, 8)), np.array([0]))
    assert result.empty


def test_compute_zone_embedding_features_valid() -> None:
    from src.nlp.embeddings import compute_zone_embedding_features

    rng = np.random.default_rng(42)
    n = 8
    embeddings = rng.standard_normal((n, 16)).astype(np.float32)
    cluster_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    df = pd.DataFrame({"zone_id": ["z1"] * 4 + ["z2"] * 4})
    result = compute_zone_embedding_features(df, embeddings, cluster_labels)
    assert not result.empty
    assert "zone_id" in result.columns
    assert "embedding_diversity" in result.columns


def test_compute_zone_embedding_features_single_row_per_zone() -> None:
    from src.nlp.embeddings import compute_zone_embedding_features

    embeddings = np.array([[1.0, 0.0, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6]], dtype=np.float32)
    cluster_labels = np.array([0])
    df = pd.DataFrame({"zone_id": ["z1"]})
    result = compute_zone_embedding_features(df, embeddings, cluster_labels)
    assert not result.empty


# ── topic model ────────────────────────────────────────────────────────────────


def test_starter_topic_labels() -> None:
    from src.nlp.topic_model import starter_topic_labels

    labels = starter_topic_labels()
    assert isinstance(labels, tuple)
    assert len(labels) >= 3
    assert "healthy" in labels


def test_discover_topics_without_texts() -> None:
    from src.nlp.topic_model import discover_topics

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((10, 8)).astype(np.float32)
    result = discover_topics(embeddings, n_topics=3)
    assert "cluster_labels" in result
    assert "topic_terms" in result
    assert "centroids" in result
    assert len(result["cluster_labels"]) == 10


def test_discover_topics_with_texts() -> None:
    from src.nlp.topic_model import discover_topics

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((6, 8)).astype(np.float32)
    texts = [
        "healthy salad",
        "fresh wrap",
        "ramen bowl",
        "tacos fresh",
        "pizza slice",
        "burger",
    ]
    result = discover_topics(embeddings, n_topics=2, texts=texts)
    assert "topic_terms" in result
    assert len(result["topic_terms"]) == 2
    for terms in result["topic_terms"].values():
        assert isinstance(terms, list)


def test_discover_topics_more_topics_than_samples() -> None:
    from src.nlp.topic_model import discover_topics

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((3, 8)).astype(np.float32)
    result = discover_topics(embeddings, n_topics=10)
    assert len(result["cluster_labels"]) == 3


def test_discover_topics_zero_n_topics_clamps_to_one() -> None:
    """Covers line 40: n_topics = 1 when min(0, N) < 1."""
    from src.nlp.topic_model import discover_topics

    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((5, 4)).astype(np.float32)
    result = discover_topics(embeddings, n_topics=0)
    assert len(result["cluster_labels"]) == 5
    assert len(result["topic_terms"]) == 1


def test_discover_topics_empty_cluster_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 57-58: empty mask for a cluster_id when all samples in one cluster."""
    import numpy as np
    from sklearn import cluster as sklearn_cluster
    from src.nlp.topic_model import discover_topics

    class _FakeKMeans:
        def __init__(self, n_clusters: int = 2, **kwargs: object) -> None:
            self.n_clusters = n_clusters
            self.cluster_centers_ = np.zeros((n_clusters, 4))

        def fit_predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

    monkeypatch.setattr(sklearn_cluster, "KMeans", _FakeKMeans)

    embeddings = np.random.default_rng(3).standard_normal((3, 4)).astype(np.float32)
    texts = ["apple", "banana", "cherry"]
    result = discover_topics(embeddings, n_topics=2, texts=texts)
    assert result["topic_terms"][1] == []


def test_topic_distribution_per_zone_empty() -> None:
    from src.nlp.topic_model import topic_distribution_per_zone

    result = topic_distribution_per_zone(pd.DataFrame(), np.zeros((0, 8)), np.array([]))
    assert result.empty


def test_topic_distribution_per_zone_no_zone_id() -> None:
    from src.nlp.topic_model import topic_distribution_per_zone

    df = pd.DataFrame({"text": ["hello"]})
    result = topic_distribution_per_zone(df, np.zeros((1, 8)), np.array([0]))
    assert result.empty


def test_topic_distribution_per_zone_valid() -> None:
    from src.nlp.topic_model import topic_distribution_per_zone

    df = pd.DataFrame({"zone_id": ["z1", "z1", "z2", "z2", "z2"]})
    embeddings = np.zeros((5, 8))
    cluster_labels = np.array([0, 1, 0, 0, 1])
    result = topic_distribution_per_zone(df, embeddings, cluster_labels)
    assert not result.empty
    assert "zone_id" in result.columns
    # should have topic share columns
    assert any("topic_" in c for c in result.columns)


# ── sentiment ─────────────────────────────────────────────────────────────────


def test_allowed_sentiment_labels() -> None:
    from src.nlp.sentiment import allowed_sentiment_labels

    labels = allowed_sentiment_labels()
    assert "positive" in labels
    assert "negative" in labels
    assert "neutral" in labels
    assert isinstance(labels, tuple)


# ── embeddings — optimal_k_search and cluster_stability ──────────────────────


def test_optimal_k_search_returns_best_k_wide_embedding() -> None:
    from src.nlp.embeddings import optimal_k_search

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, 16)).astype(np.float32)
    best_k, scores = optimal_k_search(embeddings, k_range=range(2, 5))
    assert best_k >= 2
    assert isinstance(scores, dict)


def test_optimal_k_search_skips_k_exceeding_samples() -> None:
    from src.nlp.embeddings import optimal_k_search

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((4, 8)).astype(np.float32)
    best_k, scores = optimal_k_search(embeddings, k_range=range(2, 10))
    assert best_k >= 2


def test_cluster_stability_returns_float() -> None:
    from src.nlp.embeddings import cluster_stability

    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal((20, 8)).astype(np.float32)
    score = cluster_stability(embeddings, n_clusters=3, n_runs=3)
    assert 0.0 <= score <= 1.0


# ── gemini labels — cache helpers ────────────────────────────────────────────


def test_gemini_load_cache_adds_rationale_column(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.nlp.gemini_labels as gl

    cache_df = pd.DataFrame(
        {
            "review_id": ["r1"],
            "sentiment": ["positive"],
            "concept_subtype": ["salad_bowls"],
            "confidence": [0.9],
        }
    )
    cache_path = tmp_path / "gemini_cache.parquet"
    cache_df.to_parquet(cache_path, index=False)
    monkeypatch.setattr(gl, "_CACHE_PATH", cache_path)
    result = gl._load_cache()
    assert result is not None
    assert "r1" in result or len(result) == 1


def test_gemini_save_cache_empty_list_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.nlp.gemini_labels as gl

    wrote = []

    def _mock_write(path, content):
        wrote.append(path)

    monkeypatch.setattr("pathlib.Path.write_bytes", _mock_write)
    gl._save_cache([])
    assert len(wrote) == 0


# ── topic model — cluster with no members ────────────────────────────────────


def test_discover_topics_with_real_word_texts() -> None:
    from src.nlp.topic_model import discover_topics

    rng = np.random.default_rng(99)
    embeddings = rng.standard_normal((6, 4)).astype(np.float32)
    texts = [
        "fresh salad vegetable",
        "spicy ramen noodle soup",
        "tacos tortilla beef",
        "pizza margherita cheese",
        "sushi salmon rice",
        "burger beef fries",
    ]
    result = discover_topics(embeddings, n_topics=3, texts=texts)
    assert "topic_terms" in result
    for terms in result["topic_terms"].values():
        assert isinstance(terms, list)


# ── review aggregates (additional) ────────────────────────────────────────────


def test_aggregate_review_labels_missing_required_columns() -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    df = pd.DataFrame({"sentiment": ["positive"], "zone_id": ["z1"]})
    result = aggregate_review_labels(df)
    assert result.empty


def test_aggregate_review_labels_all_low_confidence() -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    df = pd.DataFrame(
        {
            "review_id": ["1"],
            "sentiment": ["positive"],
            "concept_subtype": ["salad_bowls"],
            "confidence": [0.3],
            "zone_id": ["z1"],
            "time_key": [2022],
        }
    )
    result = aggregate_review_labels(df)
    assert result.empty


def test_aggregate_review_labels_include_sentiment_dist(
    sample_review_labels: pd.DataFrame,
) -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    result = aggregate_review_labels(sample_review_labels, include_sentiment_dist=True)
    if not result.empty:
        assert "frac_positive" in result.columns
        assert "frac_negative" in result.columns


def test_aggregate_review_labels_with_topic_distribution(
    sample_review_labels: pd.DataFrame,
) -> None:
    from src.nlp.review_aggregates import aggregate_review_labels

    topic_dist = pd.DataFrame(
        {"zone_id": ["tandon-campus", "columbia-morn"], "topic_0_share": [0.6, 0.4]}
    )
    result = aggregate_review_labels(
        sample_review_labels, topic_distribution=topic_dist
    )
    assert isinstance(result, pd.DataFrame)


def test_aggregate_nlp_features_runs() -> None:
    from src.nlp.review_aggregates import aggregate_nlp_features

    rng = np.random.default_rng(42)
    n = 6
    reviews_df = pd.DataFrame(
        {"zone_id": ["z1"] * 3 + ["z2"] * 3, "review_text": ["text"] * n}
    )
    embeddings = rng.standard_normal((n, 8)).astype(np.float32)
    cluster_labels = np.array([0, 1, 0, 1, 0, 1])
    gemini_labels = pd.DataFrame(
        {
            "review_id": [str(i) for i in range(n)],
            "sentiment": ["positive"] * n,
            "concept_subtype": ["salad_bowls"] * n,
            "confidence": [0.9] * n,
            "zone_id": ["z1"] * 3 + ["z2"] * 3,
            "time_key": [2023] * n,
        }
    )
    result = aggregate_nlp_features(
        reviews_df, embeddings, cluster_labels, gemini_labels
    )
    assert isinstance(result, pd.DataFrame)


# ── gemini_labels — _load_cache and _save_cache exception paths ───────────────


def test_gemini_load_cache_exception_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import src.nlp.gemini_labels as gl

    corrupt = tmp_path / "labels.parquet"
    corrupt.write_bytes(b"not a parquet file")
    monkeypatch.setattr(gl, "_CACHE_PATH", corrupt)
    assert gl._load_cache() is None


def test_gemini_save_cache_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.nlp.gemini_labels as gl
    from src.nlp.gemini_labels import GeminiReviewLabel

    label = GeminiReviewLabel(
        review_id="abc",
        sentiment="positive",
        concept_subtype="healthy_bowl",
        confidence=0.9,
        rationale="good",
    )
    monkeypatch.setattr(gl, "_CACHE_PATH", Path("/nonexistent_dir_xyz/labels.parquet"))
    gl._save_cache([label])  # must not raise


# ── gemini_labels — cached-hit and all-cached-return paths ────────────────────


def test_gemini_label_reviews_cached_hit_and_all_cached(monkeypatch) -> None:
    import src.nlp.gemini_labels as gl
    from src.nlp.gemini_labels import (
        GeminiReviewLabel,
        _cache_key,
        label_reviews_with_gemini,
    )

    review_text = "fresh organic salad bowl"
    subtypes = ("healthy_bowl", "salad_bar")
    review_id = _cache_key(review_text, subtypes)

    cached_label = GeminiReviewLabel(
        review_id=review_id,
        sentiment="positive",
        concept_subtype="healthy_bowl",
        confidence=0.95,
        rationale="healthy",
    )
    monkeypatch.setattr(gl, "_load_cache", lambda: {review_id: cached_label})
    monkeypatch.setattr(gl, "_save_cache", lambda labels: None)

    import google.genai as genai

    class _MockClient:
        def __init__(self, api_key=None):
            pass

    monkeypatch.setattr(genai, "Client", _MockClient)

    result = label_reviews_with_gemini(
        [review_text], subtypes=subtypes, api_key="fake-key"
    )
    assert len(result) == 1
    assert result[0].sentiment == "positive"
