"""Tests for NLP helper modules."""

from src.nlp.subtype_classifier import classify_subtype
from src.nlp.white_space import compute_subtype_gap


def test_subtype_classifier_recognizes_indian_gap() -> None:
    """Subtype classification should capture healthy Indian phrasing."""

    assert classify_subtype("Looking for a healthy Indian lunch bowl option") == "healthy_indian"


def test_subtype_gap_is_non_negative() -> None:
    """The gap helper should not emit negative scores."""

    assert compute_subtype_gap(0.2, 0.5) == 0.0
