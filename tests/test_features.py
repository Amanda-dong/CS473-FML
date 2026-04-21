"""Tests for feature helpers and taxonomy."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.healthy_gap import score_healthy_gap
from src.features.microzones import default_microzones
from src.utils.taxonomy import all_known_subtypes, canonical_subtype, healthy_taxonomy


# ── taxonomy ──────────────────────────────────────────────────────────────────

def test_taxonomy_contains_healthy_indian() -> None:
    assert "healthy_indian" in healthy_taxonomy()


def test_taxonomy_contains_all_major_cuisines() -> None:
    tax = healthy_taxonomy()
    for cuisine in ("mexican", "japanese", "korean", "italian", "thai"):
        assert cuisine in tax, f"{cuisine} missing from taxonomy"


def test_taxonomy_order_healthy_indian_before_salad_bowls() -> None:
    """healthy_indian must precede salad_bowls so 'Indian bowl' classifies correctly."""
    keys = list(healthy_taxonomy().keys())
    assert keys.index("healthy_indian") < keys.index("salad_bowls")


def test_canonical_subtype_normalises_known() -> None:
    assert canonical_subtype("Healthy Indian") == "healthy_indian"
    assert canonical_subtype("salad bowls") == "salad_bowls"
    assert canonical_subtype("RAMEN") == "ramen"


def test_canonical_subtype_slugifies_unknown() -> None:
    result = canonical_subtype("Peruvian Ceviche")
    assert " " not in result
    assert result == result.lower()


def test_all_known_subtypes_is_tuple() -> None:
    assert isinstance(all_known_subtypes(), tuple)
    assert len(all_known_subtypes()) >= 5


# ── microzones ────────────────────────────────────────────────────────────────

def test_microzones_are_available() -> None:
    assert len(default_microzones()) >= 2


def test_microzones_have_valid_zone_types() -> None:
    from src.config.constants import MICROZONE_TYPES

    for z in default_microzones():
        assert z.zone_type in MICROZONE_TYPES, f"{z.zone_id} has invalid zone_type {z.zone_type}"


# ── healthy gap scoring ───────────────────────────────────────────────────────

def test_healthy_gap_scoring_returns_named_fields(sample_zone_features: dict[str, float]) -> None:
    result = score_healthy_gap(sample_zone_features)
    for field in ("healthy_gap_score", "healthy_supply_ratio", "subtype_gap", "quick_lunch_demand"):
        assert field in result


def test_healthy_gap_score_is_non_negative(sample_zone_features: dict[str, float]) -> None:
    result = score_healthy_gap(sample_zone_features)
    assert result["healthy_gap_score"] >= 0.0


def test_healthy_gap_score_zero_inputs() -> None:
    result = score_healthy_gap({})
    assert result["healthy_gap_score"] == 0.0


# ── license velocity ──────────────────────────────────────────────────────────

def test_license_velocity_computes_net_opens(sample_license_events: pd.DataFrame) -> None:
    from src.features.license_velocity import build_license_velocity_features

    result = build_license_velocity_features(sample_license_events)
    assert "license_velocity" in result.columns
    assert not result.empty


def test_license_velocity_empty_input() -> None:
    from src.features.license_velocity import build_license_velocity_features

    result = build_license_velocity_features(pd.DataFrame())
    assert list(result.columns) == ["zone_id", "time_key", "license_velocity", "net_opens", "net_closes"]


def test_license_velocity_has_zone_id(sample_license_events: pd.DataFrame) -> None:
    from src.features.license_velocity import build_license_velocity_features

    result = build_license_velocity_features(sample_license_events)
    assert "zone_id" in result.columns


# ── rent trajectory ───────────────────────────────────────────────────────────

def test_rent_trajectory_normalizes_to_unit_interval(sample_pluto_frame: pd.DataFrame) -> None:
    from src.features.rent_trajectory import build_rent_trajectory_features

    result = build_rent_trajectory_features(sample_pluto_frame)
    assert result["rent_pressure"].between(0.0, 1.0).all()


def test_rent_trajectory_empty_input() -> None:
    from src.features.rent_trajectory import build_rent_trajectory_features

    result = build_rent_trajectory_features(pd.DataFrame())
    assert "rent_pressure" in result.columns


# ── demand signals ────────────────────────────────────────────────────────────

def test_demand_signals_merges_empty_inputs() -> None:
    from src.features.demand_signals import build_demand_features

    result = build_demand_features(pd.DataFrame(), pd.DataFrame())
    assert "healthy_review_share" in result.columns


def test_demand_signals_healthy_review_share_computation() -> None:
    from src.features.demand_signals import compute_healthy_review_share

    df = pd.DataFrame({"review_text": ["great salad bowl", "average burger place", "fresh healthy wrap"]})
    share = compute_healthy_review_share(df, ["salad", "healthy", "fresh"])
    assert 0.0 < share <= 1.0


# ── feature matrix ────────────────────────────────────────────────────────────

def test_feature_matrix_joins_tables() -> None:
    from src.features.feature_matrix import build_feature_matrix

    tables = {
        "a": pd.DataFrame({"zone_id": ["z1"], "time_key": [2022], "feat_a": [1.0]}),
        "b": pd.DataFrame({"zone_id": ["z1"], "time_key": [2022], "feat_b": [2.0]}),
    }
    result = build_feature_matrix(tables)
    assert "feat_a" in result.columns
    assert "feat_b" in result.columns


def test_feature_matrix_normalize() -> None:
    from src.features.feature_matrix import normalize_feature_matrix

    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [10.0, 20.0, None]})
    result = normalize_feature_matrix(df)
    assert not result.isnull().any().any()


def test_build_zone_year_matrix_enriches_yelp_reviews_from_inspections() -> None:
    from src.features.feature_matrix import build_zone_year_matrix

    etl_outputs = {
        "yelp": pd.DataFrame(
            {
                "review_date": ["2024-03-01", "2024-04-01"],
                "business_id": ["b1", "b2"],
                "restaurant_id": ["camis-1", "camis-1"],
                "rating": [5, 3],
                "review_text": ["fresh salad bowl", "great burger"],
            }
        ),
        "inspections": pd.DataFrame(
            {
                "inspection_date": ["2024-01-15"],
                "restaurant_id": ["camis-1"],
                "grade": ["A"],
                "nta_id": ["BK09"],
            }
        ),
    }

    result = build_zone_year_matrix(etl_outputs)

    assert "healthy_review_share" in result.columns
    assert result["healthy_review_share"].notna().any()
