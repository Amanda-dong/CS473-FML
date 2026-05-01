import pandas as pd
import pytest

from src.data.enrich_zone_features import (
    _cuisine_diversity_features,
    _yelp_nta_features,
    main,
)

# ... [existing tests] ...


def test_enrich_zone_main_merges_and_saves(monkeypatch) -> None:
    # Setup mocks
    mock_zone_features = pd.DataFrame({"zone_id": ["N1"], "nta_id": ["N1"]})
    mock_inspections = pd.DataFrame({"nta_id": ["N1"], "cuisine_type": ["C1"]})
    mock_yelp = pd.DataFrame({"nta": ["N1"], "rating": [5.0]})

    mock_cuisine_df = pd.DataFrame(
        {
            "zone_id": ["N1"],
            "cuisine_diversity": [0.5],
            "dominant_cuisine": ["C1"],
            "high_risk_cuisine_share": [0.1],
        }
    )

    mock_yelp_df = pd.DataFrame(
        {
            "zone_id": ["N1"],
            "yelp_avg_rating": [5.0],
            "yelp_review_density": [1.0],
        }
    )

    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda path: (
            mock_zone_features if "zone_features" in str(path) else mock_inspections
        ),
    )
    monkeypatch.setattr("pandas.read_csv", lambda path: mock_yelp)
    monkeypatch.setattr(
        "src.data.enrich_zone_features._cuisine_diversity_features",
        lambda x: mock_cuisine_df,
    )
    monkeypatch.setattr(
        "src.data.enrich_zone_features._yelp_nta_features", lambda x: mock_yelp_df
    )
    monkeypatch.setattr("pandas.DataFrame.to_parquet", lambda self, *a, **kw: None)

    # Call main
    main()


def test_enrich_zone_main_drops_old_columns(monkeypatch) -> None:
    # Setup mocks with old columns
    mock_zone_features = pd.DataFrame(
        {
            "zone_id": ["N1"],
            "nta_id": ["N1"],
            "cuisine_diversity": [0.9],
            "dominant_cuisine": ["Old"],
        }
    )
    mock_inspections = pd.DataFrame({"nta_id": ["N1"], "cuisine_type": ["C1"]})
    mock_yelp = pd.DataFrame({"nta": ["N1"], "rating": [5.0]})

    mock_cuisine_df = pd.DataFrame(
        {
            "zone_id": ["N1"],
            "cuisine_diversity": [0.5],
            "dominant_cuisine": ["C1"],
            "high_risk_cuisine_share": [0.1],
        }
    )

    mock_yelp_df = pd.DataFrame(
        {
            "zone_id": ["N1"],
            "yelp_avg_rating": [5.0],
            "yelp_review_density": [1.0],
        }
    )

    monkeypatch.setattr(
        "pandas.read_parquet",
        lambda path: (
            mock_zone_features if "zone_features" in str(path) else mock_inspections
        ),
    )
    monkeypatch.setattr("pandas.read_csv", lambda path: mock_yelp)
    monkeypatch.setattr(
        "src.data.enrich_zone_features._cuisine_diversity_features",
        lambda x: mock_cuisine_df,
    )
    monkeypatch.setattr(
        "src.data.enrich_zone_features._yelp_nta_features", lambda x: mock_yelp_df
    )
    monkeypatch.setattr("pandas.DataFrame.to_parquet", lambda self, *a, **kw: None)

    # Call main - should not raise Duplicate column error
    main()


# ── _cuisine_diversity_features ───────────────────────────────────────────────


def test_cuisine_diversity_basic_columns() -> None:
    df = pd.DataFrame(
        {
            "nta_id": ["MN2601", "MN2601", "MN2601", "BK0900"],
            "cuisine_type": ["Chinese", "American", "Mexican", "Italian"],
        }
    )
    result = _cuisine_diversity_features(df)
    assert set(result.columns) == {
        "zone_id",
        "cuisine_diversity",
        "dominant_cuisine",
        "high_risk_cuisine_share",
    }
    assert len(result) == 2  # Two distinct NTAs


def test_cuisine_diversity_entropy_range() -> None:
    """Diversity score must be in [0, 1]."""
    df = pd.DataFrame(
        {
            "nta_id": ["MN2601"] * 6,
            "cuisine_type": [
                "Chinese",
                "American",
                "Mexican",
                "Italian",
                "Thai",
                "Greek",
            ],
        }
    )
    result = _cuisine_diversity_features(df)
    row = result[result["zone_id"] == "MN2601"].iloc[0]
    assert 0.0 <= row["cuisine_diversity"] <= 1.0


def test_cuisine_diversity_single_cuisine_nta() -> None:
    """Single-cuisine NTA: entropy = 0 but no division-by-zero (max_entropy guard)."""
    df = pd.DataFrame(
        {
            "nta_id": ["QN0100", "QN0100", "QN0100"],
            "cuisine_type": ["Italian", "Italian", "Italian"],
        }
    )
    result = _cuisine_diversity_features(df)
    assert len(result) == 1
    row = result.iloc[0]
    # Entropy is 0 for a single cuisine; normalized score should be 0 or close
    assert row["cuisine_diversity"] == pytest.approx(0.0, abs=0.01)
    assert row["dominant_cuisine"] == "italian"


def test_cuisine_diversity_high_risk_share() -> None:
    """High-risk cuisines (chinese, american, etc.) share is computed correctly."""
    df = pd.DataFrame(
        {
            "nta_id": ["BK0900"] * 4,
            "cuisine_type": ["Chinese", "Chinese", "Italian", "French"],
        }
    )
    result = _cuisine_diversity_features(df)
    row = result[result["zone_id"] == "BK0900"].iloc[0]
    # 2 of 4 rows are "chinese" (high-risk) → share = 0.5
    assert row["high_risk_cuisine_share"] == pytest.approx(0.5, abs=0.001)


def test_cuisine_diversity_filters_non_six_char_ntas() -> None:
    """Pre-2020 4-char NTA codes (e.g. 'BK09') are dropped before aggregation."""
    df = pd.DataFrame(
        {
            "nta_id": ["BK09", "BK09", "MN2601"],  # BK09 is 4-char → dropped
            "cuisine_type": ["Chinese", "Italian", "American"],
        }
    )
    result = _cuisine_diversity_features(df)
    assert len(result) == 1
    assert result.iloc[0]["zone_id"] == "MN2601"


def test_cuisine_diversity_drops_null_nta_and_cuisine() -> None:
    df = pd.DataFrame(
        {
            "nta_id": [None, "MN2601", "MN2601"],
            "cuisine_type": ["Chinese", None, "American"],
        }
    )
    result = _cuisine_diversity_features(df)
    # Only MN2601 with non-null cuisine survives
    assert len(result) == 1
    assert result.iloc[0]["zone_id"] == "MN2601"


def test_cuisine_diversity_empty_input() -> None:
    result = _cuisine_diversity_features(
        pd.DataFrame(columns=["nta_id", "cuisine_type"])
    )
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ── _yelp_nta_features ────────────────────────────────────────────────────────


def test_yelp_nta_features_basic_columns() -> None:
    df = pd.DataFrame(
        {
            "nta": ["MN2601", "MN2601", "BK0900"],
            "rating": [4.0, 3.5, 4.5],
        }
    )
    result = _yelp_nta_features(df)
    assert set(result.columns) == {"zone_id", "yelp_avg_rating", "yelp_review_density"}
    assert len(result) == 2


def test_yelp_nta_features_avg_rating_correct() -> None:
    df = pd.DataFrame(
        {
            "nta": ["MN2601", "MN2601"],
            "rating": [4.0, 5.0],
        }
    )
    result = _yelp_nta_features(df)
    row = result[result["zone_id"] == "MN2601"].iloc[0]
    assert row["yelp_avg_rating"] == pytest.approx(4.5, abs=0.001)


def test_yelp_nta_features_density_max_is_one() -> None:
    """The NTA with the most reviews gets density = 1.0."""
    df = pd.DataFrame(
        {
            "nta": ["MN2601"] * 10 + ["BK0900"] * 3,
            "rating": [4.0] * 13,
        }
    )
    result = _yelp_nta_features(df)
    max_density = result["yelp_review_density"].max()
    assert max_density == pytest.approx(1.0, abs=0.001)


def test_yelp_nta_features_density_relative() -> None:
    """NTA with 3 reviews and NTA with 6 reviews: density should be 0.5 vs 1.0."""
    df = pd.DataFrame(
        {
            "nta": ["MN2601"] * 6 + ["BK0900"] * 3,
            "rating": [4.0] * 9,
        }
    )
    result = _yelp_nta_features(df).set_index("zone_id")
    assert result.loc["MN2601", "yelp_review_density"] == pytest.approx(1.0, abs=0.001)
    assert result.loc["BK0900", "yelp_review_density"] == pytest.approx(0.5, abs=0.001)


def test_yelp_nta_features_filters_non_six_char_ntas() -> None:
    """4-char NTA codes (pre-2020) are dropped."""
    df = pd.DataFrame(
        {
            "nta": ["BK09", "MN2601"],  # BK09 is 4-char → dropped
            "rating": [4.0, 3.5],
        }
    )
    result = _yelp_nta_features(df)
    assert len(result) == 1
    assert result.iloc[0]["zone_id"] == "MN2601"


def test_yelp_nta_features_drops_null_ratings() -> None:
    df = pd.DataFrame(
        {
            "nta": ["MN2601", "MN2601", "BK0900"],
            "rating": [4.0, None, 3.5],
        }
    )
    result = _yelp_nta_features(df)
    mn_row = result[result["zone_id"] == "MN2601"].iloc[0]
    # Only the non-null rating (4.0) is used
    assert mn_row["yelp_avg_rating"] == pytest.approx(4.0, abs=0.001)


def test_yelp_nta_features_drops_non_numeric_ratings() -> None:
    df = pd.DataFrame(
        {
            "nta": ["MN2601", "MN2601"],
            "rating": ["four", 4.0],
        }
    )
    result = _yelp_nta_features(df)
    mn_row = result[result["zone_id"] == "MN2601"].iloc[0]
    assert mn_row["yelp_avg_rating"] == pytest.approx(4.0, abs=0.001)


def test_yelp_nta_features_empty_input() -> None:
    result = _yelp_nta_features(pd.DataFrame(columns=["nta", "rating"]))
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_yelp_nta_features_all_null_nta_returns_empty() -> None:
    df = pd.DataFrame({"nta": [None, None], "rating": [4.0, 3.5]})
    result = _yelp_nta_features(df)
    assert result.empty
