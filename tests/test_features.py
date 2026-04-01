"""Tests for feature helpers and taxonomy."""

from src.features.healthy_gap import score_healthy_gap
from src.features.microzones import default_microzones
from src.utils.taxonomy import healthy_taxonomy


def test_taxonomy_contains_healthy_indian() -> None:
    """Subtype-aware whitespace depends on healthy Indian support."""

    assert "healthy_indian" in healthy_taxonomy()


def test_microzones_are_available() -> None:
    """The frontend and geospatial teams need starter micro-zones."""

    assert len(default_microzones()) >= 2


def test_healthy_gap_scoring_returns_named_fields(sample_zone_features: dict[str, float]) -> None:
    """The score helper should return a stable payload shape."""

    score_payload = score_healthy_gap(sample_zone_features)
    assert "healthy_gap_score" in score_payload
