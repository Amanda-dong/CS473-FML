"""Tests for temporal validation helpers."""

import numpy as np
import pandas as pd
import pytest

from src.validation.backtesting import (
    apply_temporal_split,
    build_blocked_splits,
    evaluate_top_k,
    run_temporal_backtest,
)


def test_build_blocked_splits_returns_expected_count() -> None:
    """Rolling splits should be created from sorted time periods."""

    frame = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022]})
    splits = build_blocked_splits(
        frame, time_col="year", min_train_periods=3, test_size=1
    )
    assert len(splits) == 2


def test_apply_temporal_split_filters_periods() -> None:
    """The helper should return only rows from the requested windows."""

    frame = pd.DataFrame({"year": [2019, 2020, 2021], "value": [1, 2, 3]})
    split = build_blocked_splits(
        frame, time_col="year", min_train_periods=2, test_size=1
    )[0]
    train_frame, test_frame = apply_temporal_split(frame, "year", split)
    assert train_frame["year"].tolist() == [2019, 2020]
    assert test_frame["year"].tolist() == [2021]


def test_evaluate_top_k_is_bounded() -> None:
    """Recall-at-k helper should return values between zero and one."""

    score = evaluate_top_k(["a", "b", "c"], ["b", "d"], k=2)
    assert 0.0 <= score <= 1.0


class _CaptureTargetModel:
    fit_targets: list[list[float]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_CaptureTargetModel":  # noqa: ARG002
        type(self).fit_targets.append(y.tolist())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


def test_run_temporal_backtest_prefers_y_composite_target() -> None:
    feature_matrix = pd.DataFrame(
        {
            "zone_id": ["z1", "z2", "z3", "z4"],
            "time_key": [2020, 2020, 2021, 2021],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "zone_id": ["z1", "z2", "z3", "z4"],
            "time_key": [2020, 2020, 2021, 2021],
            "y_composite": [0.9, 0.1, 0.8, 0.2],
            "label_quality": [0.2, 0.2, 1.0, 1.0],
        }
    )

    _CaptureTargetModel.fit_targets = []
    run_temporal_backtest(
        feature_matrix=feature_matrix,
        ground_truth=ground_truth,
        model_cls=_CaptureTargetModel,
        min_train_years=1,
    )

    assert _CaptureTargetModel.fit_targets == [[0.9, 0.1]]


# ── backtesting — additional metrics ─────────────────────────────────────────


def test_ndcg_at_k_perfect_ranking() -> None:
    from src.validation.backtesting import ndcg_at_k

    preds = np.array([0.9, 0.8, 0.7])
    true = np.array([1.0, 0.5, 0.0])
    score = ndcg_at_k(preds, true, k=3)
    assert score == pytest.approx(1.0)


def test_ndcg_at_k_zero_relevance() -> None:
    from src.validation.backtesting import ndcg_at_k

    preds = np.array([0.9, 0.8, 0.7])
    true = np.array([0.0, 0.0, 0.0])
    score = ndcg_at_k(preds, true, k=3)
    assert score == 0.0


def test_expected_calibration_error_empty() -> None:
    from src.validation.backtesting import expected_calibration_error

    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_expected_calibration_error_perfect() -> None:
    from src.validation.backtesting import expected_calibration_error

    pred = np.linspace(0, 1, 100)
    actual = np.linspace(0, 1, 100)
    ece = expected_calibration_error(pred, actual)
    assert ece < 0.1


def test_bootstrap_metric_returns_ci() -> None:
    from src.validation.backtesting import bootstrap_metric

    y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y_true = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    result = bootstrap_metric(
        lambda p, t: float(np.corrcoef(p, t)[0, 1]),
        y_pred,
        y_true,
        n_bootstrap=50,
    )
    assert "mean" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["ci_lower"] <= result["ci_upper"]


def test_mean_average_precision_empty_true() -> None:
    from src.validation.backtesting import mean_average_precision

    score = mean_average_precision(["a", "b", "c"], set())
    assert score == 0.0


def test_mean_average_precision_hit() -> None:
    from src.validation.backtesting import mean_average_precision

    score = mean_average_precision(["a", "b", "c"], {"a", "b"})
    assert score > 0.0


def test_calibration_analysis_returns_dataframe() -> None:
    from src.validation.backtesting import calibration_analysis

    pred = np.linspace(0, 1, 20)
    actual = np.linspace(0, 1, 20)
    df = calibration_analysis(pred, actual, n_bins=5)
    assert not df.empty
    assert "mean_predicted" in df.columns
    assert "mean_actual" in df.columns


def test_train_test_split_by_cutoff() -> None:
    from src.validation.backtesting import train_test_split_by_cutoff

    frame = pd.DataFrame({"year": [2019, 2020, 2021, 2022], "v": [1, 2, 3, 4]})
    train, test = train_test_split_by_cutoff(frame, "year", train_end=2020, test_start=2021)
    assert train["year"].max() == 2020
    assert test["year"].min() == 2021


def test_evaluate_top_k_zero_k_raises() -> None:
    from src.validation.backtesting import evaluate_top_k

    with pytest.raises(ValueError):
        evaluate_top_k(["a"], ["b"], k=0)


def test_evaluate_top_k_no_observed() -> None:
    from src.validation.backtesting import evaluate_top_k

    assert evaluate_top_k(["a", "b"], [], k=2) == 0.0


# ── ablation ──────────────────────────────────────────────────────────────────


class _SimpleModel:
    """Minimal sklearn-compatible model for ablation tests."""

    def fit(self, X, y):
        self._coef = y.mean() if len(y) > 0 else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self._coef)


def test_feature_ablation_returns_dataframe() -> None:
    from src.validation.ablation import feature_ablation

    n = 20
    X = pd.DataFrame({"demand": np.random.default_rng(1).random(n), "rent": np.random.default_rng(2).random(n), "survival": np.random.default_rng(3).random(n)})
    y = pd.Series(np.random.default_rng(4).random(n))
    splits = [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    groups = {"demand": ["demand"], "cost": ["rent"], "viability": ["survival"]}
    result = feature_ablation(_SimpleModel, X, y, groups, splits)
    assert isinstance(result, pd.DataFrame)
    assert "ndcg_drop" in result.columns


def test_feature_ablation_skips_missing_groups() -> None:
    from src.validation.ablation import feature_ablation

    n = 10
    X = pd.DataFrame({"a": range(n)})
    y = pd.Series(range(n), dtype=float)
    splits = [(list(range(5)), list(range(5, 10)))]
    result = feature_ablation(_SimpleModel, X, y, {"nonexistent": ["missing"]}, splits)
    assert result.empty


def test_baseline_comparison_returns_all_models() -> None:
    from src.validation.ablation import baseline_comparison

    rng = np.random.default_rng(0)
    n = 15
    X = pd.DataFrame(
        {
            "quick_lunch_demand": rng.random(n),
            "subtype_gap": rng.random(n),
            "survival_score": rng.random(n),
            "rent_pressure": rng.random(n),
            "competition_score": rng.random(n),
            "healthy_review_share": rng.random(n),
        }
    )
    y = pd.Series(rng.random(n))
    splits = [(list(range(8)), list(range(8, 15)))]
    result = baseline_comparison(X, y, _SimpleModel(), splits)
    assert isinstance(result, pd.DataFrame)
    model_names = result["model_name"].tolist()
    assert "learned" in model_names
    assert "random" in model_names
    assert "popularity" in model_names
    assert "heuristic" in model_names


def test_permutation_importance_returns_per_feature() -> None:
    from src.validation.ablation import permutation_importance

    rng = np.random.default_rng(42)
    n = 10
    X = pd.DataFrame({"feat_a": rng.random(n), "feat_b": rng.random(n)})
    y = pd.Series(rng.random(n))
    model = _SimpleModel().fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=3)
    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert "importance_mean" in result.columns
    assert "importance_std" in result.columns
    assert len(result) == 2


def test_popularity_scores_demand_signal() -> None:
    from src.validation.ablation import _popularity_scores

    X = pd.DataFrame({"demand_signal": [0.5, 0.8, 0.3]})
    scores = _popularity_scores(X)
    assert len(scores) == 3
    assert scores[1] == pytest.approx(0.8)


def test_popularity_scores_quick_lunch_fallback() -> None:
    from src.validation.ablation import _popularity_scores

    X = pd.DataFrame({"quick_lunch_demand": [0.7, 0.2]})
    scores = _popularity_scores(X)
    assert scores[0] == pytest.approx(0.7)


def test_popularity_scores_default() -> None:
    from src.validation.ablation import _popularity_scores

    X = pd.DataFrame({"other": [1, 2]})
    scores = _popularity_scores(X)
    assert (scores == 0.5).all()


def test_heuristic_scores_runs() -> None:
    from src.validation.ablation import _heuristic_scores

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "quick_lunch_demand": rng.random(5),
            "subtype_gap": rng.random(5),
            "survival_score": rng.random(5),
            "rent_pressure": rng.random(5),
            "competition_score": rng.random(5),
            "healthy_review_share": rng.random(5),
            "license_velocity": rng.random(5),
        }
    )
    scores = _heuristic_scores(X)
    assert len(scores) == 5
    assert isinstance(scores, np.ndarray)


# ── backtesting — additional code paths ──────────────────────────────────────


def test_run_temporal_backtest_fallback_target_column() -> None:
    from src.validation.backtesting import run_temporal_backtest

    feature_matrix = pd.DataFrame(
        {
            "zone_id": ["z1", "z2", "z3", "z4"],
            "time_key": [2020, 2020, 2021, 2021],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "zone_id": ["z1", "z2", "z3", "z4"],
            "time_key": [2020, 2020, 2021, 2021],
            "my_target": [0.9, 0.1, 0.8, 0.2],
        }
    )

    class _SimpleModel:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))

    run_temporal_backtest(
        feature_matrix=feature_matrix,
        ground_truth=ground_truth,
        model_cls=_SimpleModel,
        min_train_years=1,
    )


def test_run_temporal_backtest_raises_no_target() -> None:
    from src.validation.backtesting import run_temporal_backtest

    feature_matrix = pd.DataFrame(
        {"zone_id": ["z1", "z2"], "time_key": [2020, 2021], "feature_a": [1.0, 2.0]}
    )
    ground_truth = pd.DataFrame(
        {"zone_id": ["z1", "z2"], "time_key": [2020, 2021]}
    )

    class _SimpleModel:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(ValueError):
        run_temporal_backtest(
            feature_matrix=feature_matrix,
            ground_truth=ground_truth,
            model_cls=_SimpleModel,
            min_train_years=1,
        )


def test_mean_average_precision_all_miss() -> None:
    from src.validation.backtesting import mean_average_precision

    score = mean_average_precision(["a", "b", "c"], {"d", "e"})
    assert score == 0.0


def test_train_test_split_by_cutoff_empty_test() -> None:
    from src.validation.backtesting import train_test_split_by_cutoff

    frame = pd.DataFrame({"year": [2018, 2019, 2020], "v": [1, 2, 3]})
    train, test = train_test_split_by_cutoff(frame, "year", train_end=2020, test_start=2021)
    assert train["year"].max() == 2020
    assert test.empty
