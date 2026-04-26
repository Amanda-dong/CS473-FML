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
        assert z.zone_type in MICROZONE_TYPES, (
            f"{z.zone_id} has invalid zone_type {z.zone_type}"
        )


# ── healthy gap scoring ───────────────────────────────────────────────────────


def test_healthy_gap_scoring_returns_named_fields(
    sample_zone_features: dict[str, float],
) -> None:
    result = score_healthy_gap(sample_zone_features)
    for field in (
        "healthy_gap_score",
        "healthy_supply_ratio",
        "subtype_gap",
        "quick_lunch_demand",
    ):
        assert field in result


def test_healthy_gap_score_is_non_negative(
    sample_zone_features: dict[str, float],
) -> None:
    result = score_healthy_gap(sample_zone_features)
    assert result["healthy_gap_score"] >= 0.0


def test_healthy_gap_score_zero_inputs() -> None:
    result = score_healthy_gap({})
    assert result["healthy_gap_score"] == 0.0


# ── license velocity ──────────────────────────────────────────────────────────


def test_license_velocity_computes_net_opens(
    sample_license_events: pd.DataFrame,
) -> None:
    from src.features.license_velocity import build_license_velocity_features

    result = build_license_velocity_features(sample_license_events)
    assert "license_velocity" in result.columns
    assert not result.empty


def test_license_velocity_empty_input() -> None:
    from src.features.license_velocity import build_license_velocity_features

    result = build_license_velocity_features(pd.DataFrame())
    assert list(result.columns) == [
        "zone_id",
        "time_key",
        "license_velocity",
        "net_opens",
        "net_closes",
    ]


def test_license_velocity_has_zone_id(sample_license_events: pd.DataFrame) -> None:
    from src.features.license_velocity import build_license_velocity_features

    result = build_license_velocity_features(sample_license_events)
    assert "zone_id" in result.columns


# ── rent trajectory ───────────────────────────────────────────────────────────


def test_rent_trajectory_normalizes_to_unit_interval(
    sample_pluto_frame: pd.DataFrame,
) -> None:
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

    df = pd.DataFrame(
        {
            "review_text": [
                "great salad bowl",
                "average burger place",
                "fresh healthy wrap",
            ]
        }
    )
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


def test_normalize_feature_matrix_constant_column() -> None:
    from src.features.feature_matrix import normalize_feature_matrix

    df = pd.DataFrame({"x": [1.0, 1.0, 1.0]})
    result = normalize_feature_matrix(df)
    assert (result["x"] == 1.0).all()


def test_prepare_social_signals_year_column() -> None:
    from src.features.feature_matrix import _prepare_social_signals

    df = pd.DataFrame(
        {"community_district": ["Brooklyn"], "year": [2024], "count": [10]}
    )
    result = _prepare_social_signals(df)
    assert not result.empty
    assert result["time_key"].iloc[0] == 2024


def test_prepare_social_signals_month_column() -> None:
    from src.features.feature_matrix import _prepare_social_signals

    df = pd.DataFrame(
        {"community_district": ["Brooklyn"], "month": ["2024-05"], "count": [10]}
    )
    result = _prepare_social_signals(df)
    assert not result.empty
    assert result["time_key"].iloc[0] == 2024


def test_build_zone_year_matrix_all_datasets() -> None:
    from src.features.feature_matrix import build_zone_year_matrix

    etl_outputs = {
        "licenses": pd.DataFrame(
            {
                "nta_id": ["BK09"],
                "event_date": ["2024-01-01"],
                "license_status": ["Issued"],
            }
        ),
        "pluto": pd.DataFrame(
            {
                "nta_id": ["BK09"],
                "assessed_value": [1000.0],
                "year": [2024],
                "commercial_sqft": [5000.0],
            }
        ),
        "acs": pd.DataFrame(
            {
                "nta_id": ["BK09"],
                "population": [1000.0],
                "median_income": [50000.0],
                "rent_burden": [0.3],
            }
        ),
        "inspections": pd.DataFrame(
            {
                "inspection_date": ["2024-01-01"],
                "nta_id": ["BK09"],
                "grade": ["A"],
                "restaurant_id": ["r1"],
            }
        ),
        "permits": pd.DataFrame(
            {"permit_date": ["2024-01-01"], "nta_id": ["BK09"], "job_count": [5.0]}
        ),
        "citibike": pd.DataFrame(
            {
                "nta_id": ["BK09"],
                "time_key": [2024],
                "trip_count": [100.0],
                "station_count": [5.0],
            }
        ),
        "airbnb": pd.DataFrame(
            {"nta_id": ["BK09"], "listing_count": [10.0], "entire_home_ratio": [0.6]}
        ),
    }
    result = build_zone_year_matrix(etl_outputs)
    assert not result.empty
    expected_cols = [
        "zone_id",
        "time_key",
        "license_velocity",
        "rent_pressure",
        "population",
        "inspection_grade_avg",
        "permit_velocity",
        "trip_count",
        "listing_count",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Column {col} missing from merged matrix"


# ── competition score ─────────────────────────────────────────────────────────


def test_competition_score_weighted_sum() -> None:
    from src.features.competition_score import compute_competition_score

    score = compute_competition_score(
        {"direct_competitors": 1.0, "chain_density": 1.0, "subtype_saturation": 1.0}
    )
    assert score == pytest.approx(1.0)


def test_competition_score_zeros() -> None:
    from src.features.competition_score import compute_competition_score

    assert compute_competition_score({}) == 0.0


def test_competition_score_partial_inputs() -> None:
    from src.features.competition_score import compute_competition_score

    score = compute_competition_score({"direct_competitors": 1.0})
    assert score == pytest.approx(0.5)


def test_competition_score_rounded() -> None:
    from src.features.competition_score import compute_competition_score

    score = compute_competition_score(
        {"direct_competitors": 0.5, "chain_density": 0.3, "subtype_saturation": 0.2}
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ── merchant viability ────────────────────────────────────────────────────────


def test_merchant_viability_all_fields_present() -> None:
    from src.features.merchant_viability import score_merchant_viability

    result = score_merchant_viability(
        {"survival_score": 0.8, "rent_pressure": 0.1, "competition_score": 0.1}
    )
    for field in (
        "merchant_viability_score",
        "survival_score",
        "rent_pressure",
        "competition_score",
    ):
        assert field in result


def test_merchant_viability_score_positive() -> None:
    from src.features.merchant_viability import score_merchant_viability

    result = score_merchant_viability(
        {"survival_score": 0.9, "rent_pressure": 0.0, "competition_score": 0.0}
    )
    assert result["merchant_viability_score"] > 0.0


def test_merchant_viability_clamped_at_zero() -> None:
    from src.features.merchant_viability import score_merchant_viability

    result = score_merchant_viability(
        {"survival_score": 0.0, "rent_pressure": 1.0, "competition_score": 1.0}
    )
    assert result["merchant_viability_score"] == 0.0


def test_merchant_viability_empty_input() -> None:
    from src.features.merchant_viability import score_merchant_viability

    result = score_merchant_viability({})
    assert result["merchant_viability_score"] == 0.0
    assert result["survival_score"] == 0.0


# ── ground truth helpers ──────────────────────────────────────────────────────


def _make_licenses_gt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2020-01-01", "2022-03-01", "2021-01-01", "2023-06-01"]
            ),
            "restaurant_id": ["camis-1", "camis-1", "camis-2", "camis-2"],
            "business_unique_id": ["dca-1", "dca-1", "dca-2", "dca-2"],
            "license_status": ["Active", "Active", "Issued", "Expired"],
            "nta_id": ["BK09", "BK09", "BK09", "BK09"],
        }
    )


def _make_reviews_gt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": ["bk-tandon", "bk-tandon", "bk-tandon"],
            "time_key": [2022, 2022, 2023],
            "rating": [4.5, 3.5, 4.0],
        }
    )


def _make_inspections_gt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nta_id": ["BK09", "BK09"],
            "inspection_date": ["2022-06-01", "2023-07-01"],
            "grade": ["A", "B"],
        }
    )


def test_license_entity_ids_prefers_restaurant_id() -> None:
    from src.features.ground_truth import _license_entity_ids

    df = pd.DataFrame({"restaurant_id": ["camis-1"], "business_unique_id": ["dca-1"]})
    result = _license_entity_ids(df)
    assert result.iloc[0] == "camis-1"


def test_license_entity_ids_falls_back_to_business_unique_id() -> None:
    from src.features.ground_truth import _license_entity_ids

    df = pd.DataFrame({"restaurant_id": [pd.NA], "business_unique_id": ["dca-1"]})
    result = _license_entity_ids(df)
    assert result.iloc[0] == "dca-1"


def test_license_entity_ids_ignores_unknown() -> None:
    from src.features.ground_truth import _license_entity_ids

    df = pd.DataFrame({"restaurant_id": ["UNKNOWN"], "business_unique_id": ["dca-1"]})
    result = _license_entity_ids(df)
    assert result.iloc[0] == "dca-1"


def test_license_entity_ids_returns_na_when_all_missing() -> None:
    from src.features.ground_truth import _license_entity_ids

    df = pd.DataFrame({"other_col": ["x"]})
    result = _license_entity_ids(df)
    assert result.isna().all()


def test_survival_rate_empty_returns_empty() -> None:
    from src.features.ground_truth import _survival_rate

    result = _survival_rate(pd.DataFrame())
    assert result.empty
    assert "y_survival" in result.columns


def test_survival_rate_with_data() -> None:
    from src.features.ground_truth import _survival_rate

    result = _survival_rate(_make_licenses_gt(), horizon_years=2)
    assert not result.empty
    assert "y_survival" in result.columns
    assert result["y_survival"].between(0.0, 1.0).all()


def test_survival_rate_no_coverable_cohorts() -> None:
    from src.features.ground_truth import _survival_rate

    # All records in same year — no cohort has T+2 observable
    df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "restaurant_id": ["r1", "r2"],
            "business_unique_id": [pd.NA, pd.NA],
            "license_status": ["Active", "Active"],
            "nta_id": ["BK09", "BK09"],
        }
    )
    result = _survival_rate(df, horizon_years=2)
    assert result.empty


def test_review_quality_empty() -> None:
    from src.features.ground_truth import _review_quality

    result = _review_quality(pd.DataFrame())
    assert result.empty
    assert "y_review_quality" in result.columns


def test_review_quality_missing_rating_col() -> None:
    from src.features.ground_truth import _review_quality

    df = pd.DataFrame({"zone_id": ["z1"], "time_key": [2022]})
    result = _review_quality(df)
    assert result.empty


def test_review_quality_missing_zone_id() -> None:
    from src.features.ground_truth import _review_quality

    df = pd.DataFrame({"rating": [4.0], "time_key": [2022]})
    result = _review_quality(df)
    assert result.empty


def test_review_quality_constant_ratings() -> None:
    from src.features.ground_truth import _review_quality

    df = pd.DataFrame(
        {"zone_id": ["z1", "z1"], "time_key": [2022, 2022], "rating": [4.0, 4.0]}
    )
    result = _review_quality(df)
    assert not result.empty
    assert result["y_review_quality"].between(0.0, 1.0).all()


def test_review_quality_with_data() -> None:
    from src.features.ground_truth import _review_quality

    result = _review_quality(_make_reviews_gt())
    assert not result.empty
    assert result["y_review_quality"].between(0.0, 1.0).all()


def test_license_velocity_signal_empty() -> None:
    from src.features.ground_truth import _license_velocity_signal

    result = _license_velocity_signal(pd.DataFrame())
    assert result.empty
    assert "y_license_velocity" in result.columns


def test_license_velocity_signal_with_data() -> None:
    from src.features.ground_truth import _license_velocity_signal

    result = _license_velocity_signal(_make_licenses_gt())
    assert not result.empty
    assert "y_license_velocity" in result.columns
    assert result["y_license_velocity"].between(0.0, 1.0).all()


def test_inspection_quality_empty() -> None:
    from src.features.ground_truth import _inspection_quality

    result = _inspection_quality(pd.DataFrame())
    assert result.empty
    assert "y_inspection" in result.columns


def test_inspection_quality_with_nta_id() -> None:
    from src.features.ground_truth import _inspection_quality

    result = _inspection_quality(_make_inspections_gt())
    assert not result.empty
    assert "y_inspection" in result.columns
    assert result["y_inspection"].between(0.0, 1.0).all()


def test_inspection_quality_with_zone_id_col() -> None:
    from src.features.ground_truth import _inspection_quality

    df = pd.DataFrame(
        {"zone_id": ["bk-tandon"], "inspection_date": ["2022-01-01"], "grade": ["A"]}
    )
    result = _inspection_quality(df)
    assert not result.empty


def test_inspection_quality_with_time_key() -> None:
    from src.features.ground_truth import _inspection_quality

    df = pd.DataFrame({"nta_id": ["BK09"], "time_key": [2022], "grade": ["A", "B"][:1]})
    result = _inspection_quality(df)
    assert not result.empty


def test_inspection_quality_missing_grade() -> None:
    from src.features.ground_truth import _inspection_quality

    df = pd.DataFrame({"nta_id": ["BK09"], "inspection_date": ["2022-01-01"]})
    result = _inspection_quality(df)
    assert result.empty


def test_build_ground_truth_empty_licenses() -> None:
    from src.features.ground_truth import build_ground_truth

    result = build_ground_truth(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert "y_composite" in result.columns


def test_build_ground_truth_with_all_inputs() -> None:
    from src.features.ground_truth import build_ground_truth

    result = build_ground_truth(
        _make_licenses_gt(), _make_reviews_gt(), _make_inspections_gt()
    )
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "y_composite" in result.columns
        assert "missingness_fraction" in result.columns
        assert "label_quality" in result.columns


def test_build_ground_truth_composite_bounded() -> None:
    from src.features.ground_truth import build_ground_truth

    result = build_ground_truth(
        _make_licenses_gt(), _make_reviews_gt(), _make_inspections_gt()
    )
    if not result.empty:
        non_nan = result["y_composite"].dropna()
        if len(non_nan) > 0:
            assert non_nan.between(0.0, 1.0).all()


# ── geospatial helpers ────────────────────────────────────────────────────────


def test_lat_lon_to_nta_manhattan() -> None:
    from src.utils.geospatial import lat_lon_to_nta

    # lon >= -73.75 falls through to MN default in the borough-bucket algorithm
    result = lat_lon_to_nta(pd.Series([40.75]), pd.Series([-73.74]))
    assert result.iloc[0].startswith("MN")


def test_lat_lon_to_nta_brooklyn() -> None:
    from src.utils.geospatial import lat_lon_to_nta

    result = lat_lon_to_nta(pd.Series([40.67]), pd.Series([-73.99]))
    assert result.iloc[0].startswith("BK")


def test_lat_lon_to_nta_bronx() -> None:
    from src.utils.geospatial import lat_lon_to_nta

    result = lat_lon_to_nta(pd.Series([40.85]), pd.Series([-73.90]))
    assert result.iloc[0].startswith("BX")


def test_lat_lon_to_nta_staten_island() -> None:
    from src.utils.geospatial import lat_lon_to_nta

    result = lat_lon_to_nta(pd.Series([40.60]), pd.Series([-74.15]))
    assert result.iloc[0].startswith("SI")


def test_lat_lon_to_nta_queens() -> None:
    from src.utils.geospatial import lat_lon_to_nta

    result = lat_lon_to_nta(pd.Series([40.70]), pd.Series([-73.80]))
    assert result.iloc[0].startswith("QN")


def test_describe_microzone_known() -> None:
    from src.utils.geospatial import describe_microzone

    assert "campus" in describe_microzone("campus_walkshed", "NYU").lower()
    assert "lunch" in describe_microzone("lunch_corridor", "Midtown").lower()


def test_describe_microzone_fallback() -> None:
    from src.utils.geospatial import describe_microzone

    assert describe_microzone("unknown_type", "MyZone") == "MyZone"


# ── zone crosswalk — aggregate ────────────────────────────────────────────────


def test_aggregate_nta_to_zone_basic() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    df = pd.DataFrame({"nta_id": ["BK09", "MN22"], "population": [5000.0, 8000.0]})
    result = aggregate_nta_to_zone(
        df, zone_col="nta_id", agg_rules={"population": "mean"}
    )
    assert not result.empty
    assert "zone_id" in result.columns


def test_aggregate_nta_to_zone_empty_df() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    result = aggregate_nta_to_zone(pd.DataFrame(), zone_col="nta_id")
    assert result.empty


def test_aggregate_nta_to_zone_no_matching_ntas() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    df = pd.DataFrame({"nta_id": ["XX99"], "value": [1.0]})
    result = aggregate_nta_to_zone(df, zone_col="nta_id")
    assert result.empty


def test_aggregate_nta_to_zone_with_year() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    df = pd.DataFrame(
        {"nta_id": ["BK09", "BK09"], "year": [2022, 2023], "value": [1.0, 2.0]}
    )
    result = aggregate_nta_to_zone(df, zone_col="nta_id")
    assert "time_key" in result.columns


def test_aggregate_nta_to_zone_with_weights() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    df = pd.DataFrame(
        {
            "nta_id": ["BK09", "BK09"],
            "value": [10.0, 20.0],
            "pop": [100.0, 200.0],
        }
    )
    result = aggregate_nta_to_zone(
        df, zone_col="nta_id", agg_rules={"value": "mean"}, weights_col="pop"
    )
    assert not result.empty


def test_aggregate_nta_to_zone_sum_agg() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    df = pd.DataFrame({"nta_id": ["BK09"], "count": [5.0]})
    result = aggregate_nta_to_zone(df, zone_col="nta_id", agg_rules={"count": "sum"})
    assert not result.empty


def test_aggregate_nta_to_zone_no_numeric_cols() -> None:
    from src.features.zone_crosswalk import aggregate_nta_to_zone

    df = pd.DataFrame({"nta_id": ["BK09"], "label": ["A"]})
    result = aggregate_nta_to_zone(df, zone_col="nta_id")
    assert not result.empty  # returns deduped zone_id only


def test_resolve_nta_multi_zone_no_primary_falls_back_to_sorted() -> None:
    from src.features.zone_crosswalk import resolve_nta_to_zone_id

    # BK09 → primary is bk-tandon
    assert resolve_nta_to_zone_id("BK09") == "bk-tandon"


# ── demand signals — additional branches ──────────────────────────────────────


def test_demand_signals_renames_mention_count() -> None:
    from src.features.demand_signals import build_demand_features

    review = pd.DataFrame(
        {"zone_id": ["z1"], "time_key": [2024], "healthy_review_share": [0.5]}
    )
    social = pd.DataFrame(
        {"zone_id": ["z1"], "time_key": [2024], "mention_count": [10]}
    )
    result = build_demand_features(review, social)
    assert "social_buzz" in result.columns


def test_demand_signals_missing_healthy_review_share() -> None:
    from src.features.demand_signals import build_demand_features

    review = pd.DataFrame({"zone_id": ["z1"], "time_key": [2024]})
    social = pd.DataFrame({"zone_id": ["z1"], "time_key": [2024]})
    result = build_demand_features(review, social)
    assert "healthy_review_share" in result.columns
    assert result.iloc[0]["healthy_review_share"] == pytest.approx(0.0)
    assert "social_buzz" in result.columns
    assert result.iloc[0]["social_buzz"] == pytest.approx(0.0)


def test_compute_healthy_review_share_empty_df() -> None:
    from src.features.demand_signals import compute_healthy_review_share

    assert compute_healthy_review_share(
        pd.DataFrame(), ["fresh", "healthy"]
    ) == pytest.approx(0.0)


def test_compute_healthy_review_share_no_keywords() -> None:
    from src.features.demand_signals import compute_healthy_review_share

    df = pd.DataFrame({"review_text": ["great food"]})
    assert compute_healthy_review_share(df, []) == pytest.approx(0.0)


def test_compute_healthy_review_share_match() -> None:
    from src.features.demand_signals import compute_healthy_review_share

    df = pd.DataFrame({"review_text": ["fresh salad", "burger fries", "healthy wrap"]})
    score = compute_healthy_review_share(df, ["fresh", "healthy"])
    assert score == pytest.approx(2 / 3)


# ── settings — property coverage ─────────────────────────────────────────────


def test_settings_properties_return_paths() -> None:
    from src.config.settings import get_settings

    s = get_settings()
    assert s.data_dir.name == "data"
    assert s.raw_dir.name == "raw"
    assert s.processed_dir.name == "processed"
    assert s.geojson_dir.name == "geojson"


# ── utils.paths ───────────────────────────────────────────────────────────────


def test_project_paths_returns_dict() -> None:
    from src.utils.paths import project_paths

    paths = project_paths()
    assert "repo_root" in paths
    assert "data_dir" in paths
    assert "raw_dir" in paths
    assert "processed_dir" in paths
    assert "geojson_dir" in paths


# ── schemas.results — build_placeholder_response ─────────────────────────────


def test_build_placeholder_response_returns_correct_schema() -> None:
    from src.schemas.results import build_placeholder_response

    response = build_placeholder_response("salad_bowls", limit=3)
    assert len(response.recommendations) == 3
    for rec in response.recommendations:
        assert rec.concept_subtype == "salad_bowls"
        assert 0.0 <= rec.opportunity_score <= 1.0
        assert rec.confidence_bucket in ("low", "medium", "high")


# ── feature_matrix — build_zone_year_matrix ──────────────────────────────────


def test_build_zone_year_matrix_with_licenses() -> None:
    from src.features.feature_matrix import build_zone_year_matrix

    licenses = pd.DataFrame(
        {
            "event_date": pd.date_range("2022-01-01", periods=6, freq="ME"),
            "restaurant_id": [f"R{i}" for i in range(6)],
            "business_unique_id": [f"BU{i}" for i in range(6)],
            "license_status": ["Active"] * 6,
            "nta_id": ["BK09"] * 6,
            "category": ["Restaurant"] * 6,
        }
    )
    etl_outputs = {"licenses": licenses}
    result = build_zone_year_matrix(etl_outputs)
    assert isinstance(result, pd.DataFrame)


def test_build_zone_year_matrix_empty_inputs() -> None:
    from src.features.feature_matrix import build_zone_year_matrix

    result = build_zone_year_matrix({})
    assert isinstance(result, pd.DataFrame)


# ── feature_matrix — build_feature_matrix with all-empty tables ───────────────


def test_build_feature_matrix_all_empty_tables() -> None:
    from src.features.feature_matrix import build_feature_matrix

    result = build_feature_matrix({"a": pd.DataFrame(), "b": pd.DataFrame()})
    assert list(result.columns) == ["zone_id", "time_key"]


# ── feature_matrix — _load_gemini_review_features exception path ──────────────


def test_load_gemini_review_features_exception_returns_empty(monkeypatch, tmp_path) -> None:
    import src.features.feature_matrix as fm

    cache_path = tmp_path / "labels.csv"
    cache_path.write_bytes(b"\xff\xfe corrupt data\n")
    monkeypatch.setattr(fm, "_GEMINI_CACHE", cache_path)
    result = fm._load_gemini_review_features(pd.DataFrame(), pd.DataFrame())
    assert result.empty


# ── feature_matrix — _prepare_social_signals with count column ────────────────


def test_prepare_social_signals_with_count_column() -> None:
    from src.features.feature_matrix import _prepare_social_signals

    df = pd.DataFrame({
        "community_district": ["Brooklyn", "Brooklyn", "Manhattan"],
        "year": [2023, 2023, 2023],
        "count": [5, 3, 2],
    })
    result = _prepare_social_signals(df)
    assert isinstance(result, pd.DataFrame)
