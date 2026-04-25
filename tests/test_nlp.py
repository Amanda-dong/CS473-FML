"""Tests for NLP helper modules."""

from __future__ import annotations

import json
import sys
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


def test_label_reviews_requires_api_key() -> None:
    """Verify labeling raises when no API key is provided."""
    from src.nlp.gemini_labels import label_reviews_with_gemini

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
