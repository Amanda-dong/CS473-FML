"""Tests for model helpers — scoring, ranking, clustering, survival."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cmf_score import LearnedScoringModel, ScoreComponents, compute_opening_score, score_zone_for_concept
from src.models.model_loader import load_feature_matrix, load_scoring_model
from src.models.ranking_model import rank_zones
from src.models.trajectory_model import TrajectoryClusteringModel


class DummyPredictor:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=float)


# ── opening score ─────────────────────────────────────────────────────────────

def test_opening_score_is_numeric() -> None:
    score = compute_opening_score(
        ScoreComponents(
            healthy_gap_score=0.8,
            subtype_gap_score=0.7,
            merchant_viability_score=0.6,
            competition_penalty=0.2,
        )
    )
    assert isinstance(score, float)


def test_opening_score_range() -> None:
    """Score should be non-negative for reasonable inputs."""
    score = compute_opening_score(
        ScoreComponents(
            healthy_gap_score=0.5,
            subtype_gap_score=0.5,
            merchant_viability_score=0.5,
            competition_penalty=0.5,
        )
    )
    assert score >= 0.0


def test_higher_gap_increases_score() -> None:
    low = compute_opening_score(ScoreComponents(0.2, 0.2, 0.5, 0.1))
    high = compute_opening_score(ScoreComponents(0.9, 0.9, 0.5, 0.1))
    assert high > low


def test_score_zone_for_concept(sample_zone_features: dict[str, float]) -> None:
    components = score_zone_for_concept(sample_zone_features, "healthy_indian")
    score = compute_opening_score(components)
    assert 0.0 <= score <= 2.0


# ── ranking ───────────────────────────────────────────────────────────────────

def test_ranking_orders_descending() -> None:
    rows = [{"zone_name": "A", "opportunity_score": 0.2}, {"zone_name": "B", "opportunity_score": 0.9}]
    ranked = rank_zones(rows)
    assert ranked[0]["zone_name"] == "B"


def test_ranking_single_row() -> None:
    rows = [{"zone_name": "Only", "opportunity_score": 0.5}]
    ranked = rank_zones(rows)
    assert len(ranked) == 1


def test_ranking_empty() -> None:
    assert rank_zones([]) == []


# ── trajectory clustering ─────────────────────────────────────────────────────

def test_trajectory_model_predicts_after_fit() -> None:
    model = TrajectoryClusteringModel().fit(pd.DataFrame({"value": [1, 2, 3]}))
    assert len(model.predict(pd.DataFrame({"value": [1, 2]}))) == 2


def test_trajectory_model_gmm() -> None:
    model = TrajectoryClusteringModel(algorithm="gmm", n_clusters=2)
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    labels = model.fit_predict(data)
    assert len(labels) == 5


def test_trajectory_model_cluster_count() -> None:
    model = TrajectoryClusteringModel(n_clusters=4)
    data = pd.DataFrame({"x": range(20), "y": range(20)})
    model.fit(data)
    assert len(model.cluster_labels_) == 4


def test_trajectory_model_describe_clusters() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model = TrajectoryClusteringModel(n_clusters=2).fit(data)
    desc = model.describe_clusters(data)
    assert not desc.empty


def test_trajectory_model_raises_before_fit() -> None:
    model = TrajectoryClusteringModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"x": [1.0]}))


# ── survival model ────────────────────────────────────────────────────────────

def test_survival_model_fits_on_test_data(sample_restaurant_history: pd.DataFrame) -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    model.fit(sample_restaurant_history)
    assert model.fitted_


def test_survival_model_predict_risk(sample_restaurant_history: pd.DataFrame) -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle().fit(sample_restaurant_history)
    risk = model.predict_risk(sample_restaurant_history.head(5))
    assert (risk >= 0.0).all() and (risk <= 1.0).all()


def test_survival_model_heuristic_fallback() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    model.fit(pd.DataFrame())  # empty → heuristic
    assert model.uses_heuristic_

    candidate = pd.DataFrame({"rent_pressure": [0.4], "competition_score": [0.3]})
    risk = model.predict_risk(candidate)
    assert 0.0 <= float(risk.iloc[0]) <= 1.0


def test_survival_synthetic_builder_removed() -> None:
    """Verify synthetic builder raises RuntimeError after removal."""
    from src.models.survival_model import build_synthetic_restaurant_history

    with pytest.raises(RuntimeError, match="Synthetic data generation removed"):
        build_synthetic_restaurant_history(n=100)


# ── explainability ────────────────────────────────────────────────────────────

def test_explainability_returns_strings(sample_zone_features: dict[str, float]) -> None:
    from src.models.explainability import top_positive_drivers, top_risks

    drivers = top_positive_drivers(sample_zone_features)
    risks = top_risks(sample_zone_features)
    assert isinstance(drivers, list) and len(drivers) > 0
    assert isinstance(risks, list) and len(risks) > 0
    assert all(isinstance(d, str) for d in drivers)
    assert all(isinstance(r, str) for r in risks)


def test_explainability_zero_features() -> None:
    from src.models.explainability import top_positive_drivers, top_risks

    drivers = top_positive_drivers({})
    risks = top_risks({})
    # Should return fallback strings, not crash
    assert len(drivers) > 0
    assert len(risks) > 0


def test_load_feature_matrix_tries_multiple_candidate_paths(tmp_path) -> None:
    frame = pd.DataFrame({"zone_id": ["z1"], "time_key": [2024], "target": [0.8]})
    path = tmp_path / "feature_matrix.parquet"
    frame.to_parquet(path, index=False)

    loaded = load_feature_matrix((tmp_path / "missing.parquet", path))

    assert loaded is not None
    assert loaded["zone_id"].tolist() == ["z1"]


def test_load_scoring_model_rehydrates_learned_wrapper(tmp_path) -> None:
    model = LearnedScoringModel(params={"n_estimators": 1})
    model.model = DummyPredictor()
    model.feature_names = ["feature_a"]
    path = tmp_path / "scoring_model.joblib"
    model.save(str(path))

    loaded = load_scoring_model(path)

    assert isinstance(loaded, LearnedScoringModel)
    assert loaded.feature_names == ["feature_a"]
    np.testing.assert_allclose(
        loaded.predict(pd.DataFrame({"feature_a": [3.0, 4.0]})),
        np.array([1.0, 1.0]),
    )


def test_build_real_restaurant_history_uses_business_unique_id_without_inspection_join() -> None:
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "restaurant_id": [pd.NA, pd.NA],
            "business_unique_id": ["dca-1", "dca-1"],
            "license_status": ["Active", "Expired"],
            "nta_id": ["BK09", "BK09"],
        }
    )
    inspections = pd.DataFrame({"restaurant_id": ["camis-1"], "grade": ["A"]})

    history = build_real_restaurant_history(licenses, inspections)

    assert history["restaurant_id"].tolist() == ["dca-1"]
    assert history["inspection_grade_numeric"].tolist() == [2.0]


# ── learned ranker ────────────────────────────────────────────────────────────

def test_learned_ranker_raises_before_fit() -> None:
    from src.models.ranking_model import LearnedRanker
    ranker = LearnedRanker()
    with pytest.raises(RuntimeError, match=r"fit\(\) before predict\(\)"):
        ranker.predict(pd.DataFrame({"x": [1]}))


def test_learned_ranker_fit_predict(tmp_path) -> None:
    from src.models.ranking_model import LearnedRanker, HAS_XGB, HAS_JOBLIB
    if not HAS_XGB or not HAS_JOBLIB:
        pytest.skip("xgboost and joblib required for LearnedRanker test")

    X = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [4, 3, 2, 1]})
    y = pd.Series([0.1, 0.4, 0.7, 0.9])
    group = [4]

    ranker = LearnedRanker(params={"n_estimators": 2, "max_depth": 1})
    ranker.fit(X, y, group)

    preds = ranker.predict(X)
    assert len(preds) == 4

    path = str(tmp_path / "ranker.joblib")
    ranker.save(path)

    loaded = LearnedRanker.load(path)
    assert loaded.feature_names == ["feat1", "feat2"]
    assert len(loaded.predict(X)) == 4


# ── survival evaluation ───────────────────────────────────────────────────────

def test_survival_model_rsf(sample_restaurant_history: pd.DataFrame) -> None:
    from src.models.survival_model import SurvivalModelBundle, HAS_SKSURV
    if not HAS_SKSURV:
        pytest.skip("sksurv required for RSF test")
    model = SurvivalModelBundle(baseline="rsf")
    model.fit(sample_restaurant_history)
    assert model.fitted_
    if not model.uses_heuristic_:
        assert model.rsf_model_ is not None
        risk = model.predict_risk(sample_restaurant_history.head(5))
        assert len(risk) == 5
        median = model.predict_median_survival(sample_restaurant_history.head(5))
        assert len(median) == 5


def test_survival_model_evaluation(sample_restaurant_history: pd.DataFrame) -> None:
    from src.models.survival_model import SurvivalModelBundle
    model = SurvivalModelBundle(baseline="cox")
    model.fit(sample_restaurant_history)

    if not model.uses_heuristic_:
        c_index = model.concordance_index(sample_restaurant_history)
        assert 0.0 <= c_index <= 1.0

        brier = model.brier_score(sample_restaurant_history, times=[100, 365])
        assert not brier.empty
        assert "brier_score" in brier.columns

        calib = model.calibration_data(sample_restaurant_history)
        assert "predicted_survival" in calib.columns

        ph_test = model.test_proportional_hazards(sample_restaurant_history)
        assert "error" not in ph_test


def test_survival_model_predict_median_survival_heuristic() -> None:
    from src.models.survival_model import SurvivalModelBundle
    model = SurvivalModelBundle()
    model.fit(pd.DataFrame())  # heuristic
    median = model.predict_median_survival(pd.DataFrame({"rent_pressure": [0.5], "competition_score": [0.5]}))
    assert float(median.iloc[0]) > 0
