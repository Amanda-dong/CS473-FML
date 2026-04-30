"""Tests for model helpers — scoring, ranking, clustering, survival."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cmf_score import (
    LearnedScoringModel,
    ScoreComponents,
    compute_opening_score,
    score_zone_for_concept,
)
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
    rows = [
        {"zone_name": "A", "opportunity_score": 0.2},
        {"zone_name": "B", "opportunity_score": 0.9},
    ]
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
    data = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )
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


def test_survival_model_fits_on_test_data(
    sample_restaurant_history: pd.DataFrame,
) -> None:
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


def test_build_real_restaurant_history_uses_business_unique_id_without_inspection_join() -> (
    None
):
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "restaurant_id": [pd.NA, pd.NA],
            "business_unique_id": ["dca-1", "dca-1"],
            "license_status": ["Active", "Expired"],
            "nta_id": ["BK0202", "BK0202"],
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
    median = model.predict_median_survival(
        pd.DataFrame({"rent_pressure": [0.5], "competition_score": [0.5]})
    )
    assert float(median.iloc[0]) > 0


# ── survival model — edge cases ───────────────────────────────────────────────


def test_survival_model_heuristic_baseline_type(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle(baseline="heuristic")
    model.fit(sample_restaurant_history)
    assert model.fitted_
    assert model.uses_heuristic_


def test_survival_model_no_rent_pressure_in_candidate() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    model.fit(pd.DataFrame())
    risk = model.predict_risk(pd.DataFrame({"other_col": [1.0, 2.0]}))
    assert len(risk) == 2


def test_survival_model_predict_risk_raises_before_fit() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    with pytest.raises(RuntimeError):
        model.predict_risk(pd.DataFrame({"rent_pressure": [0.5]}))


def test_survival_model_predict_median_raises_before_fit() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    with pytest.raises(RuntimeError):
        model.predict_median_survival(pd.DataFrame({"rent_pressure": [0.5]}))


def test_survival_model_concordance_index_raises_before_fit() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    with pytest.raises(RuntimeError):
        model.concordance_index(pd.DataFrame())


def test_survival_model_brier_score_raises_before_fit() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    with pytest.raises(RuntimeError):
        model.brier_score(pd.DataFrame(), times=[100])


def test_survival_model_calibration_data_raises_before_fit() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    with pytest.raises(RuntimeError):
        model.calibration_data(pd.DataFrame())


def test_survival_model_ph_test_without_cox() -> None:
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    model.fit(pd.DataFrame())  # heuristic → no cox model
    result = model.test_proportional_hazards(pd.DataFrame())
    assert "error" in result


def test_build_real_restaurant_history_empty_licenses() -> None:
    from src.models.survival_model import build_real_restaurant_history

    empty_licenses = pd.DataFrame(
        columns=[
            "event_date",
            "restaurant_id",
            "business_unique_id",
            "license_status",
            "nta_id",
        ]
    )
    result = build_real_restaurant_history(empty_licenses, pd.DataFrame())
    assert "restaurant_id" in result.columns
    assert result.empty


def test_build_real_restaurant_history_with_closed_status() -> None:
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2022-06-01"]),
            "restaurant_id": ["camis-1", "camis-1"],
            "business_unique_id": [pd.NA, pd.NA],
            "license_status": ["Active", "Expired"],
            "nta_id": ["BK0202", "BK0202"],
        }
    )
    inspections = pd.DataFrame({"restaurant_id": ["camis-1"], "grade": ["A"]})
    result = build_real_restaurant_history(licenses, inspections)
    assert not result.empty
    assert result.iloc[0]["event_observed"] == 1


def test_build_real_restaurant_history_no_restaurant_id_col() -> None:
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01"]),
            "business_unique_id": ["dca-99"],
            "license_status": ["Active"],
            "nta_id": ["MN17"],
        }
    )
    result = build_real_restaurant_history(licenses, pd.DataFrame())
    assert "restaurant_id" in result.columns


def test_build_real_restaurant_history_with_zone_features() -> None:
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2019-01-01", "2021-01-01"]),
            "restaurant_id": ["r1", "r1"],
            "business_unique_id": [pd.NA, pd.NA],
            "license_status": ["Active", "Active"],
            "nta_id": ["BK0202", "BK0202"],
        }
    )
    zone_features = pd.DataFrame(
        {
            "zone_id": ["bk-tandon"],
            "rent_pressure": [0.3],
            "competition_score": [0.2],
            "transit_access": [0.7],
        }
    )
    result = build_real_restaurant_history(licenses, pd.DataFrame(), zone_features)
    assert not result.empty


# ── baselines ─────────────────────────────────────────────────────────────────


def test_build_baseline_runs_returns_list() -> None:
    from src.models.baselines import build_baseline_runs

    runs = build_baseline_runs()
    assert len(runs) >= 3


def test_build_baseline_runs_have_required_fields() -> None:
    from src.models.baselines import BaselineRun, build_baseline_runs

    runs = build_baseline_runs()
    for run in runs:
        assert isinstance(run, BaselineRun)
        assert run.name
        assert run.owner
        assert run.notes


# ── trajectory model — additional ─────────────────────────────────────────────


def test_trajectory_model_auto_select_k_kmeans() -> None:
    data = pd.DataFrame({"x": range(15), "y": range(15)})
    model = TrajectoryClusteringModel(n_clusters=None, algorithm="kmeans")
    model.fit(data)
    assert model.fitted_
    assert "n_clusters" in model.diagnostics_


def test_trajectory_model_auto_select_k_gmm() -> None:
    data = pd.DataFrame({"x": range(15), "y": range(15)})
    model = TrajectoryClusteringModel(n_clusters=None, algorithm="gmm")
    model.fit(data)
    assert model.fitted_
    assert "bic" in model.diagnostics_


def test_trajectory_model_cluster_stability_runs() -> None:
    data = pd.DataFrame({"x": range(20), "y": range(20)})
    model = TrajectoryClusteringModel(n_clusters=3).fit(data)
    scaled = model.scaler_.transform(data.select_dtypes(include=["number"]).fillna(0.0))
    ari = model.cluster_stability(scaled, n_runs=3)
    assert 0.0 <= ari <= 1.0


def test_trajectory_model_sweep_k() -> None:
    data = pd.DataFrame({"x": range(20), "y": range(20)})
    model = TrajectoryClusteringModel(n_clusters=3).fit(data)
    sweep = model.sweep_k(data, k_range=range(2, 4))
    assert not sweep.empty
    assert "k" in sweep.columns
    assert "silhouette" in sweep.columns


def test_trajectory_model_sweep_k_gmm() -> None:
    data = pd.DataFrame({"x": range(20), "y": range(20)})
    model = TrajectoryClusteringModel(algorithm="gmm", n_clusters=2).fit(data)
    sweep = model.sweep_k(data, k_range=range(2, 4))
    assert not sweep.empty
    assert "bic" in sweep.columns


def test_trajectory_model_gmm_diagnostics() -> None:
    data = pd.DataFrame({"a": range(10), "b": range(10)})
    model = TrajectoryClusteringModel(algorithm="gmm", n_clusters=2).fit(data)
    assert "bic" in model.diagnostics_
    assert "aic" in model.diagnostics_


def test_trajectory_model_inertia_diagnostics() -> None:
    data = pd.DataFrame({"a": range(10), "b": range(10)})
    model = TrajectoryClusteringModel(algorithm="kmeans", n_clusters=2).fit(data)
    assert "inertia" in model.diagnostics_


def test_trajectory_model_empty_raises() -> None:
    with pytest.raises(ValueError):
        TrajectoryClusteringModel().fit(pd.DataFrame({"label": ["a", "b"]}))


# ── cmf_score — additional branches ──────────────────────────────────────────


def test_learned_scoring_model_predict_with_model() -> None:
    model = LearnedScoringModel()
    model.model = DummyPredictor()
    model.feature_names = ["f"]
    preds = model.predict(pd.DataFrame({"f": [1.0, 2.0]}))
    assert len(preds) == 2


def test_learned_scoring_model_predict_raises_without_fit() -> None:
    model = LearnedScoringModel()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict(pd.DataFrame({"f": [1.0]}))


def test_learned_scoring_model_predict_uncertainty_raises_without_fit() -> None:
    model = LearnedScoringModel()
    with pytest.raises(RuntimeError):
        model.predict_with_uncertainty(pd.DataFrame({"f": [1.0]}))


def test_learned_scoring_model_explain_raises_without_fit() -> None:
    model = LearnedScoringModel()
    with pytest.raises(RuntimeError):
        model.explain(pd.DataFrame({"f": [1.0]}))


def test_learned_scoring_model_save_requires_joblib(monkeypatch, tmp_path) -> None:
    import src.models.cmf_score as cmf_module

    monkeypatch.setattr(cmf_module, "HAS_JOBLIB", False)
    m = LearnedScoringModel()
    with pytest.raises(ImportError, match="joblib"):
        m.save(str(tmp_path / "m.joblib"))


def test_learned_scoring_model_load_requires_joblib(monkeypatch) -> None:
    import src.models.cmf_score as cmf_module

    monkeypatch.setattr(cmf_module, "HAS_JOBLIB", False)
    with pytest.raises(ImportError, match="joblib"):
        LearnedScoringModel.load("any_path")


def test_learned_scoring_model_fit_requires_xgboost(monkeypatch) -> None:
    import src.models.cmf_score as cmf_module

    monkeypatch.setattr(cmf_module, "HAS_XGB", False)
    m = LearnedScoringModel()
    with pytest.raises(ImportError, match="xgboost"):
        m.fit(pd.DataFrame({"f": [1.0]}), pd.Series([1.0]))


def test_learned_scoring_model_fit_with_eval_set() -> None:
    """Covers lines 177-178: eval_set is passed through to XGBRegressor.fit()."""
    n = 10
    X = pd.DataFrame({"a": range(n), "b": range(n)}, dtype=float)
    y = pd.Series(range(n), dtype=float)
    X_val = pd.DataFrame({"a": [0.0], "b": [0.0]})
    y_val = pd.Series([0.0])
    model = LearnedScoringModel(params={"n_estimators": 2})
    model.fit(X, y, eval_set=[(X_val, y_val)])
    assert model.feature_names == ["a", "b"]


def test_learned_scoring_model_explain_requires_shap(monkeypatch) -> None:
    import src.models.cmf_score as cmf_module

    monkeypatch.setattr(cmf_module, "HAS_SHAP", False)
    m = LearnedScoringModel()
    m.model = DummyPredictor()
    with pytest.raises(ImportError, match="shap"):
        m.explain(pd.DataFrame({"f": [1.0]}))


def test_learned_scoring_model_load_classmethod(tmp_path) -> None:
    import joblib

    data = {"model": DummyPredictor(), "feature_names": ["feat_x"], "params": {}}
    path = tmp_path / "direct_load.joblib"
    joblib.dump(data, path)
    loaded = LearnedScoringModel.load(str(path))
    assert loaded.feature_names == ["feat_x"]


def test_score_zone_for_concept_vel_norm_zero() -> None:
    """When license_velocity is exactly 0.0, sigmoid returns 0.5."""
    components = score_zone_for_concept({"license_velocity": 0.0}, "healthy_indian")
    assert components.license_velocity_score == pytest.approx(0.5)


# ── explainability — additional branches ─────────────────────────────────────


def test_top_positive_drivers_transit_access() -> None:
    from src.models.explainability import top_positive_drivers

    features = {"transit_access": 0.9, "income_alignment": 0.8}
    drivers = top_positive_drivers(features)
    assert any("transit" in d.lower() for d in drivers)


def test_top_positive_drivers_income_alignment() -> None:
    from src.models.explainability import top_positive_drivers

    features = {"income_alignment": 0.8}
    drivers = top_positive_drivers(features)
    assert any("income" in d.lower() for d in drivers)


def test_top_risks_competition_and_rent() -> None:
    from src.models.explainability import top_risks

    features = {"competition_score": 0.8, "rent_pressure": 0.8, "survival_score": 0.2}
    risks = top_risks(features)
    assert len(risks) >= 2
    assert any("competition" in r.lower() or "rent" in r.lower() for r in risks)


def test_top_risks_income_and_transit() -> None:
    from src.models.explainability import top_risks

    features = {"income_alignment": 0.1, "transit_access": 0.2}
    risks = top_risks(features)
    assert any("income" in r.lower() or "transit" in r.lower() for r in risks)


def test_shap_drivers_with_mock_model() -> None:
    from src.models.explainability import shap_drivers

    class MockExplainableModel:
        def explain(self, df):
            vals = pd.DataFrame(
                {"feat_a": [0.5], "feat_b": [-0.3], "feat_c": [0.1]}, index=df.index
            )
            return vals

    model = MockExplainableModel()
    row = pd.Series({"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3.0})
    positives, risks = shap_drivers(model, row, top_n=2)
    assert isinstance(positives, list)
    assert isinstance(risks, list)
    assert len(positives) == 2
    assert len(risks) == 2


# ── model_loader — additional ─────────────────────────────────────────────────


def test_save_model_creates_meta_json(tmp_path) -> None:
    from src.models.model_loader import save_model, get_model_metadata

    model = LearnedScoringModel()
    model.feature_names = ["a", "b"]
    path = tmp_path / "test_model.joblib"
    save_model(model, path, metadata={"training_rows": 100})
    meta = get_model_metadata(path)
    assert meta is not None
    assert "saved_at" in meta
    assert "model_type" in meta
    assert meta["training_rows"] == 100


def test_get_model_version_unknown(tmp_path) -> None:
    from src.models.model_loader import get_model_version

    version = get_model_version(tmp_path / "nonexistent.joblib")
    assert version == "unknown"


def test_get_model_version_with_meta(tmp_path) -> None:
    from src.models.model_loader import save_model, get_model_version

    model = LearnedScoringModel()
    path = tmp_path / "versioned.joblib"
    save_model(model, path)
    version = get_model_version(path)
    assert "LearnedScoringModel" in version


def test_load_survival_model_returns_none_when_missing(tmp_path) -> None:
    from src.models.model_loader import load_survival_model

    result = load_survival_model(tmp_path / "missing_surv.joblib")
    assert result is None


def test_load_survival_model_from_disk(tmp_path) -> None:
    import joblib
    from src.models.model_loader import load_survival_model
    from src.models.survival_model import SurvivalModelBundle

    model = SurvivalModelBundle()
    model.fit(pd.DataFrame())
    path = tmp_path / "surv.joblib"
    joblib.dump(model, path)
    loaded = load_survival_model(path)
    assert loaded is not None


def test_load_feature_matrix_returns_none_when_missing(tmp_path) -> None:
    from src.models.model_loader import load_feature_matrix

    result = load_feature_matrix(tmp_path / "missing_matrix.parquet")
    assert result is None


def test_candidate_paths_single() -> None:
    from src.models.model_loader import _candidate_paths

    paths = _candidate_paths("some/path.joblib")
    assert len(paths) == 1


# ── survival model — additional evaluation methods ────────────────────────────


def test_survival_model_predict_median_survival_cox(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle()
    bundle.fit(sample_restaurant_history)
    result = bundle.predict_median_survival(sample_restaurant_history.head(5))
    assert len(result) == 5
    assert (result > 0).all()


def test_survival_model_concordance_index_runs(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle()
    bundle.fit(sample_restaurant_history)
    c = bundle.concordance_index(sample_restaurant_history)
    assert 0.0 <= c <= 1.0


def test_survival_model_brier_score_returns_dataframe(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle()
    bundle.fit(sample_restaurant_history)
    result = bundle.brier_score(sample_restaurant_history, times=[365, 730])
    assert isinstance(result, pd.DataFrame)
    assert "brier_score" in result.columns


def test_survival_model_calibration_data_returns_dataframe(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle()
    bundle.fit(sample_restaurant_history)
    result = bundle.calibration_data(sample_restaurant_history)
    assert isinstance(result, pd.DataFrame)


def test_survival_model_test_ph_no_cox() -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle(baseline="heuristic")
    bundle.fit(pd.DataFrame())
    result = bundle.test_proportional_hazards(pd.DataFrame())
    assert "error" in result


def test_survival_model_test_ph_exception(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle()
    bundle.fit(sample_restaurant_history)
    if bundle.cox_model_ is not None:
        original = bundle.cox_model_.check_assumptions

        def _raise(*a, **kw):
            raise RuntimeError("intentional error")

        bundle.cox_model_.check_assumptions = _raise
        result = bundle.test_proportional_hazards(sample_restaurant_history)
        assert "error" in result
        bundle.cox_model_.check_assumptions = original


def test_survival_model_fit_zero_variance_uses_heuristic() -> None:
    from src.models.survival_model import SurvivalModelBundle

    history = pd.DataFrame(
        {
            "duration_days": [100, 200, 300, 400, 500] * 4,
            "event_observed": [1, 0, 1, 0, 1] * 4,
            "constant_a": [1.0] * 20,
            "constant_b": [2.0] * 20,
        }
    )
    bundle = SurvivalModelBundle()
    bundle.fit(history)
    assert bundle.uses_heuristic_


def test_survival_model_no_entity_id_col() -> None:
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2022-03-01"]),
            "license_status": ["Active", "Expired"],
            "nta_id": ["BK0202", "BK0202"],
        }
    )
    result = build_real_restaurant_history(licenses, pd.DataFrame())
    assert "restaurant_id" in result.columns


# ── cmf_score — predict_with_uncertainty ──────────────────────────────────────


def test_learned_scoring_model_predict_with_uncertainty() -> None:
    from src.models.cmf_score import LearnedScoringModel

    model = LearnedScoringModel()
    X = pd.DataFrame({"a": [0.5, 0.6, 0.7], "b": [0.1, 0.2, 0.3]})
    y = pd.Series([0.9, 0.8, 0.7])
    model.fit(X, y)
    mean_pred, ci_lower, ci_upper = model.predict_with_uncertainty(X, n_bootstrap=5)
    assert mean_pred.shape == (3,)
    assert (ci_lower <= ci_upper).all()


# ── ranking_model — save and load ────────────────────────────────────────────


def test_learned_ranker_save_requires_joblib(monkeypatch) -> None:
    import src.models.ranking_model as rm_module
    from src.models.ranking_model import LearnedRanker

    monkeypatch.setattr(rm_module, "HAS_JOBLIB", False)
    r = LearnedRanker()
    with pytest.raises(ImportError, match="joblib"):
        r.save("some/path.joblib")


def test_learned_ranker_load_requires_joblib(monkeypatch) -> None:
    import src.models.ranking_model as rm_module
    from src.models.ranking_model import LearnedRanker

    monkeypatch.setattr(rm_module, "HAS_JOBLIB", False)
    with pytest.raises(ImportError, match="joblib"):
        LearnedRanker.load("some/path.joblib")


def test_learned_ranker_save_and_load_roundtrip(tmp_path) -> None:
    import joblib
    from src.models.ranking_model import LearnedRanker

    path = tmp_path / "ranker.joblib"
    data = {"model": None, "feature_names": ["x", "y"], "params": {}}
    joblib.dump(data, path)
    loaded = LearnedRanker.load(str(path))
    assert loaded.feature_names == ["x", "y"]


# ── model_loader — load feature matrix success ────────────────────────────────


def test_load_feature_matrix_success(tmp_path) -> None:
    from src.models.model_loader import load_feature_matrix

    df = pd.DataFrame({"zone_id": ["z1", "z2"], "val": [1.0, 2.0]})
    path = tmp_path / "matrix.parquet"
    df.to_parquet(path)
    loaded = load_feature_matrix(path)
    assert loaded is not None
    assert len(loaded) == 2


def test_save_model_with_sklearn_feature_names(tmp_path) -> None:
    import types
    from src.models.model_loader import save_model, get_model_metadata

    model = types.SimpleNamespace(feature_names_=["feat1", "feat2"])
    path = tmp_path / "sklearn_model.joblib"
    save_model(model, path)
    meta = get_model_metadata(path)
    assert meta is not None
    assert "feature_names" in meta


# ── trajectory model — sweep_k and edge cases ────────────────────────────────


def test_trajectory_model_sweep_k_runs(sample_restaurant_history: pd.DataFrame) -> None:
    from src.models.trajectory_model import TrajectoryClusteringModel

    model = TrajectoryClusteringModel(n_clusters=2)
    model.fit(sample_restaurant_history)
    result = model.sweep_k(sample_restaurant_history, k_range=range(2, 4))
    assert isinstance(result, pd.DataFrame)
    assert "k" in result.columns
    assert "silhouette" in result.columns


# ── model_loader — exception handlers ─────────────────────────────────────────


def test_save_model_exception_is_swallowed(tmp_path, monkeypatch) -> None:
    import joblib
    import src.models.model_loader as ml

    monkeypatch.setattr(
        joblib, "dump", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
    )
    ml.save_model(object(), tmp_path / "bad.joblib")  # must not raise


@pytest.mark.parametrize(
    "fn_name,suffix",
    [
        ("load_scoring_model", "joblib"),
        ("load_survival_model", "joblib"),
        ("load_feature_matrix", "parquet"),
    ],
)
def test_load_model_corrupt_file_returns_none(
    tmp_path, fn_name: str, suffix: str
) -> None:
    import src.models.model_loader as ml

    p = tmp_path / f"bad.{suffix}"
    p.write_bytes(b"not a valid file")
    assert getattr(ml, fn_name)(p) is None


# ── survival_model — partial-variance drop and Cox convergence failure ─────────


def test_survival_model_drops_near_zero_variance_cols() -> None:
    from src.models.survival_model import SurvivalModelBundle

    df = pd.DataFrame(
        {
            "duration_days": [100, 200, 300, 400, 500],
            "event_observed": [1, 0, 1, 0, 1],
            "feature_constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            "feature_var": [0.1, 0.5, 0.9, 0.3, 0.7],
        }
    )
    bundle = SurvivalModelBundle()
    bundle.fit(df)
    assert "feature_var" in bundle.feature_columns_
    assert "feature_constant" not in bundle.feature_columns_


def test_survival_model_cox_convergence_failure(monkeypatch) -> None:
    import src.models.survival_model as sm_module

    class _FailCox:
        def __init__(self, **kwargs):
            pass

        def fit(self, *a, **kw):
            raise RuntimeError("convergence failure")

    monkeypatch.setattr(sm_module, "CoxPHFitter", _FailCox)
    df = pd.DataFrame(
        {
            "duration_days": [100, 200, 300],
            "event_observed": [1, 0, 1],
            "feature_var": [0.1, 0.5, 0.9],
        }
    )
    bundle = sm_module.SurvivalModelBundle()
    bundle.fit(df)
    assert bundle.uses_heuristic_
    assert bundle.cox_model_ is None


# ── survival_model — brier_score alt paths ────────────────────────────────────


def test_survival_model_brier_score_heuristic_path(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle(baseline="heuristic")
    bundle.fit(sample_restaurant_history)
    result = bundle.brier_score(sample_restaurant_history, times=[365])
    assert isinstance(result, pd.DataFrame)
    assert "brier_score" in result.columns


def test_survival_model_brier_score_zero_informative() -> None:
    from src.models.survival_model import SurvivalModelBundle

    df = pd.DataFrame(
        {
            "duration_days": [10, 20],
            "event_observed": [0, 0],
            "rent_pressure": [0.5, 0.3],
        }
    )
    bundle = SurvivalModelBundle(baseline="heuristic")
    bundle.fit(df)
    result = bundle.brier_score(df, times=[9999])
    assert np.isnan(result["brier_score"].iloc[0])


def test_survival_model_brier_score_no_ipcw_fallback(
    monkeypatch,
    sample_restaurant_history: pd.DataFrame,
) -> None:
    import lifelines
    from src.models.survival_model import SurvivalModelBundle

    class _BadKMF:
        def fit(self, *a, **kw):
            raise RuntimeError("forced")

    monkeypatch.setattr(lifelines, "KaplanMeierFitter", _BadKMF)
    bundle = SurvivalModelBundle(baseline="heuristic")
    bundle.fit(sample_restaurant_history)
    result = bundle.brier_score(sample_restaurant_history, times=[365])
    assert isinstance(result, pd.DataFrame)


# ── survival_model — calibration_data alt paths ───────────────────────────────


def test_survival_model_calibration_data_heuristic_path(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    bundle = SurvivalModelBundle(baseline="heuristic")
    bundle.fit(sample_restaurant_history)
    result = bundle.calibration_data(sample_restaurant_history)
    assert isinstance(result, pd.DataFrame)


def test_survival_model_calibration_data_empty_informative() -> None:
    from src.models.survival_model import SurvivalModelBundle

    df = pd.DataFrame(
        {
            "duration_days": [10],
            "event_observed": [0],
            "rent_pressure": [0.5],
        }
    )
    bundle = SurvivalModelBundle(baseline="heuristic")
    bundle.fit(df)
    result = bundle.calibration_data(df, horizon_days=9999)
    assert result.empty


# ── ranking_model — fit without xgboost ──────────────────────────────────────


def test_learned_ranker_fit_requires_xgboost(monkeypatch) -> None:
    import src.models.ranking_model as rm_module
    from src.models.ranking_model import LearnedRanker

    monkeypatch.setattr(rm_module, "HAS_XGB", False)
    r = LearnedRanker()
    with pytest.raises(ImportError, match="xgboost"):
        r.fit(pd.DataFrame({"f": [1.0]}), pd.Series([1.0]), group=[1])


# ── trajectory_model — GMM stability and auto k_range ─────────────────────────


def test_trajectory_model_gmm_cluster_stability(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.trajectory_model import TrajectoryClusteringModel

    model = TrajectoryClusteringModel(n_clusters=2, algorithm="gmm")
    model.fit(sample_restaurant_history)
    scaled = model.scaler_.transform(
        sample_restaurant_history[model.feature_columns_].fillna(0.0)
    )
    score = model.cluster_stability(scaled, n_runs=3)
    assert isinstance(score, float)


def test_trajectory_model_sweep_k_auto_range(
    sample_restaurant_history: pd.DataFrame,
) -> None:
    from src.models.trajectory_model import TrajectoryClusteringModel

    model = TrajectoryClusteringModel(n_clusters=2)
    model.fit(sample_restaurant_history)
    result = model.sweep_k(sample_restaurant_history)
    assert isinstance(result, pd.DataFrame)
    assert "k" in result.columns


# ── survival_model — coverage gaps ──────────────────────────────────────────


def test_survival_model_rsf_fallback_to_cox(sample_restaurant_history, monkeypatch):
    from src.models import survival_model
    from src.models.survival_model import SurvivalModelBundle

    monkeypatch.setattr(survival_model, "HAS_SKSURV", False)
    monkeypatch.setattr(survival_model, "HAS_LIFELINES", True)

    model = SurvivalModelBundle(baseline="rsf")
    model.fit(sample_restaurant_history)

    assert model.fitted_
    assert model.uses_heuristic_ is False
    assert model.cox_model_ is not None


def test_survival_model_fit_heuristic_no_lifelines(
    sample_restaurant_history, monkeypatch
):
    from src.models import survival_model
    from src.models.survival_model import SurvivalModelBundle

    monkeypatch.setattr(survival_model, "HAS_LIFELINES", False)

    model = SurvivalModelBundle(baseline="cox")
    model.fit(sample_restaurant_history)

    assert model.fitted_
    assert model.uses_heuristic_


def test_survival_model_fit_cox_early_return(monkeypatch):
    from src.models import survival_model
    from src.models.survival_model import SurvivalModelBundle

    monkeypatch.setattr(survival_model, "HAS_LIFELINES", False)

    model = SurvivalModelBundle()
    model.feature_columns_ = ["feat"]
    frame = pd.DataFrame({"duration_days": [10], "event_observed": [1], "feat": [1]})
    model._fit_cox(frame)

    assert model.uses_heuristic_


def test_survival_model_fit_rsf_mocked(monkeypatch):
    from src.models import survival_model
    from src.models.survival_model import SurvivalModelBundle

    class FakeRSF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True

    monkeypatch.setattr(survival_model, "HAS_SKSURV", True)
    monkeypatch.setattr(survival_model, "RandomSurvivalForest", FakeRSF, raising=False)

    model = SurvivalModelBundle(baseline="rsf")
    model.feature_columns_ = ["rent_pressure", "competition_score"]
    model_frame = pd.DataFrame(
        {
            "duration_days": [10, 20],
            "event_observed": [1, 0],
            "rent_pressure": [0.5, 0.6],
            "competition_score": [0.3, 0.4],
        }
    )

    model._fit_rsf(model_frame)
    assert model.rsf_model_ is not None
    assert model.rsf_model_.fitted


def test_survival_model_predict_risk_rsf_mocked():
    from src.models.survival_model import SurvivalModelBundle

    class MockFunc:
        def __init__(self, y):
            self.y = y

    class MockRSF:
        def predict_cumulative_hazard_function(self, X):
            return [MockFunc(np.array([0.0, 0.3, 0.8]))]

    model = SurvivalModelBundle()
    model.fitted_ = True
    model.uses_heuristic_ = False
    model.feature_columns_ = ["rent_pressure"]
    model.rsf_model_ = MockRSF()

    candidate = pd.DataFrame({"rent_pressure": [0.5]})
    risk = model.predict_risk(candidate)

    assert 0.0 <= float(risk.iloc[0]) <= 1.0


def test_survival_model_predict_median_survival_rsf_mocked():
    from src.models.survival_model import SurvivalModelBundle

    class MockFunc:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockRSF:
        def predict_survival_function(self, X):
            return [MockFunc(np.array([30, 365, 730]), np.array([0.9, 0.4, 0.1]))]

    model = SurvivalModelBundle()
    model.fitted_ = True
    model.uses_heuristic_ = False
    model.feature_columns_ = ["rent_pressure"]
    model.rsf_model_ = MockRSF()

    candidate = pd.DataFrame({"rent_pressure": [0.5]})
    median = model.predict_median_survival(candidate)

    assert float(median.iloc[0]) == 365.0


def test_trajectory_model_sweep_k_single_cluster():
    from src.models.trajectory_model import TrajectoryClusteringModel

    data = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]})
    model = TrajectoryClusteringModel(n_clusters=1)
    sweep = model.sweep_k(data, k_range=range(1, 2))

    assert (sweep["silhouette"] == -1.0).any()


def test_build_real_restaurant_history_zone_features_join():
    from src.models.survival_model import build_real_restaurant_history

    licenses = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01"]),
            "business_unique_id": ["BU1"],
            "license_status": ["Active"],
            "nta_id": ["BK0202"],
        }
    )
    zone_features = pd.DataFrame(
        {
            "zone_id": ["BK0202"],
            "rent_pressure": [0.4],
            "competition_score": [0.3],
            "transit_access": [0.6],
        }
    )

    result = build_real_restaurant_history(
        licenses, pd.DataFrame(), zone_features=zone_features
    )
    assert "rent_pressure" in result.columns
    assert result.iloc[0]["rent_pressure"] == 0.4
    assert result.iloc[0]["competition_score"] == 0.3
    assert result.iloc[0]["transit_access"] == 0.6


def test_survival_model_rsf_fit_monkeypatch(
    monkeypatch, sample_restaurant_history
) -> None:
    from src.models.survival_model import SurvivalModelBundle

    class FakeRSF:
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, X, y):
            self.fitted_ = True
            return self

    monkeypatch.setattr(
        "src.models.survival_model.RandomSurvivalForest", FakeRSF, raising=False
    )
    monkeypatch.setattr("src.models.survival_model.HAS_SKSURV", True)

    bundle = SurvivalModelBundle(baseline="rsf")
    bundle.fit(sample_restaurant_history)

    assert bundle.rsf_model_ is not None


def test_trajectory_find_best_k_degenerate(monkeypatch) -> None:
    from src.models.trajectory_model import TrajectoryClusteringModel
    import sklearn.cluster

    # Monkeypatch fit_predict to return all zeros (degenerate clusters)
    def fake_fit_predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    monkeypatch.setattr(sklearn.cluster.KMeans, "fit_predict", fake_fit_predict)

    model = TrajectoryClusteringModel(algorithm="kmeans", n_clusters=None)
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    # _find_best_k is internal, but we can call it or call fit()
    # n_clusters=None triggers _find_best_k in fit()
    model.fit(data)
    # If all k in k_range [2, 3] (default) fail silhouette_score because labels same,
    # it continues and returns 2.
    # Note: silhouette_score raises ValueError if len(set(labels)) < 2,
    # but the code has a check for that.
