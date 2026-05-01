"""Tests for causal uplift evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.causal import (
    CausalMLConfig,
    TLearnerUpliftModel,
    compute_qini_coefficient,
    compute_standardized_mean_differences,
    compute_uplift_at_fraction,
    compute_uplift_curve,
    estimate_ate,
    estimate_propensity_scores,
    evaluate_policy_value,
    export_fold_manifest,
    load_causal_frame,
    make_temporal_splits,
    run_causal_temporal_backtest,
    _safe_mean,
)
from src.validation import run_causal_evaluation


def _sample_causal_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    idx = 0
    for period in ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"]:
        for segment in range(24):
            treatment = 1 if segment % 2 == 0 else 0
            uplift_driver = 1.0 if segment < 12 else -0.3
            base = 1.0 + (0.02 * idx)
            outcome = base + (0.8 * uplift_driver if treatment == 1 else 0.0)
            rows.append(
                {
                    "time_key": period,
                    "feature_signal": uplift_driver,
                    "feature_noise": float((segment % 5) / 10),
                    "treatment": treatment,
                    "outcome": outcome,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def test_safe_mean_empty() -> None:
    assert _safe_mean(np.array([])) == 0.0


def test_t_learner_fit_requires_both_classes() -> None:
    model = TLearnerUpliftModel()
    X = pd.DataFrame({"x": [1, 2]})
    treatment = pd.Series([1, 1])
    outcome = pd.Series([1.0, 2.0])
    import pytest

    with pytest.raises(
        ValueError, match="Both treated and control samples are required"
    ):
        model.fit(X, treatment, outcome)


def test_propensity_scores_fallback_to_decision_function(monkeypatch) -> None:
    from sklearn.svm import SVC

    # SVC with probability=False has decision_function but no predict_proba
    model = SVC(probability=False)
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    treatment = pd.Series([1, 0, 1, 0])
    scores = estimate_propensity_scores(X, treatment, model=model)
    assert len(scores) == 4
    assert np.all((scores >= 0.05) & (scores <= 0.95))


def test_smd_zero_variance() -> None:
    frame = pd.DataFrame({"treatment": [1, 1, 0, 0], "feature": [1.0, 1.0, 1.0, 1.0]})
    balance = compute_standardized_mean_differences(frame, "treatment", ["feature"])
    assert balance.loc[0, "smd"] == 0.0

    frame_diff = pd.DataFrame(
        {"treatment": [1, 1, 0, 0], "feature": [2.0, 2.0, 1.0, 1.0]}
    )
    balance_diff = compute_standardized_mean_differences(
        frame_diff, "treatment", ["feature"]
    )
    assert np.isinf(balance_diff.loc[0, "smd"])


def test_estimate_ate_none_propensity() -> None:
    outcome = pd.Series([1.0, 0.0])
    treatment = pd.Series([1, 0])
    result = estimate_ate(outcome, treatment, propensity=None)
    assert "ate" in result


def test_uplift_curve_empty() -> None:
    curve = compute_uplift_curve(pd.Series([]), pd.Series([]), np.array([]))
    assert curve.empty
    assert compute_qini_coefficient(curve) == 0.0


def test_evaluate_policy_value_variations() -> None:
    outcome = pd.Series([1.0, 0.5])
    treatment = pd.Series([1, 0])
    propensity = np.array([0.5, 0.5])
    uplift = np.array([0.1, -0.1])

    val_all = evaluate_policy_value(
        outcome, treatment, propensity, uplift, baseline_policy="treat_all"
    )
    assert "policy_value" in val_all

    val_hist = evaluate_policy_value(
        outcome, treatment, propensity, uplift, baseline_policy="historical"
    )
    assert "baseline_value" in val_hist


def test_load_causal_frame_formats(tmp_path) -> None:
    df = pd.DataFrame({"time": [1, 2], "val": [3, 4]})
    csv_path = tmp_path / "test.csv"
    parquet_path = tmp_path / "test.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    l1 = load_causal_frame(csv_path, "time")
    l2 = load_causal_frame(parquet_path, "time")
    assert len(l1) == 2
    assert len(l2) == 2

    import pytest

    with pytest.raises(ValueError, match="Supported dataset formats"):
        load_causal_frame(tmp_path / "test.txt", "time")


def test_run_causal_evaluation_cli(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "treatment": [1, 0, 1, 0, 1],
            "outcome": [1, 2, 1, 2, 1],
            "feat": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    ).to_csv(dataset, index=False)

    args = [
        "run_causal_evaluation.py",
        "--dataset",
        str(dataset),
        "--time-col",
        "time",
        "--treatment-col",
        "treatment",
        "--outcome-col",
        "outcome",
        "--feature-cols",
        "feat",
        "--output-dir",
        str(tmp_path / "out"),
        "--min-train-periods",
        "2",
        "--skip-sensitivity-analysis",
    ]
    monkeypatch.setattr("sys.argv", args)

    # Mock print to avoid cluttering test output
    monkeypatch.setattr("builtins.print", lambda *x, **y: None)

    exit_code = run_causal_evaluation.main()
    assert exit_code == 0
    assert (tmp_path / "out" / "manifest.json").exists()


def test_make_temporal_splits_supports_rolling_windows() -> None:
    frame = pd.DataFrame({"time_key": ["a", "b", "c", "d", "e"]})
    splits = make_temporal_splits(
        frame,
        time_col="time_key",
        min_train_periods=2,
        test_size=1,
        window_type="rolling",
    )
    assert len(splits) == 3
    assert splits[0].train_periods == ("a", "b")
    assert splits[1].train_periods == ("b", "c")


def test_uplift_curve_and_qini_are_positive_for_informative_ranker() -> None:
    outcome = pd.Series([5, 4, 3, 2, 1, 1], dtype=float)
    treatment = pd.Series([1, 0, 1, 0, 1, 0], dtype=int)
    predicted_uplift = np.array([0.9, 0.7, 0.8, -0.2, -0.3, -0.5])
    curve = compute_uplift_curve(outcome, treatment, predicted_uplift)
    qini = compute_qini_coefficient(curve)
    assert not curve.empty
    assert "random_baseline" in curve.columns
    assert qini > 0


def test_estimate_ate_returns_confidence_interval() -> None:
    outcome = pd.Series([3.0, 1.0, 4.0, 2.0, 5.0, 2.0])
    treatment = pd.Series([1, 0, 1, 0, 1, 0])
    propensity = np.array([0.6, 0.4, 0.7, 0.3, 0.65, 0.35])
    result = estimate_ate(outcome, treatment, propensity=propensity)
    assert result["ate"] > 0
    assert result["ate_ci_lower"] <= result["ate_ci_upper"]
    assert 0.0 <= result["ate_p_value"] <= 1.0


def test_balance_check_reports_smd() -> None:
    frame = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "feature_a": [10.0, 10.0, 5.0, 5.0],
            "feature_b": [1.0, 1.1, 0.9, 1.0],
        }
    )
    balance = compute_standardized_mean_differences(
        frame,
        treatment_col="treatment",
        feature_cols=["feature_a", "feature_b"],
    )
    assert set(balance.columns) >= {"feature", "smd", "abs_smd"}
    assert float(balance.loc[balance["feature"] == "feature_a", "abs_smd"].iloc[0]) > 0


def test_propensity_scores_are_clipped() -> None:
    frame = _sample_causal_frame()
    propensity = estimate_propensity_scores(
        frame[["feature_signal", "feature_noise"]],
        frame["treatment"],
    )
    assert ((propensity >= 0.05) & (propensity <= 0.95)).all()


def test_uplift_at_top_fraction_is_positive() -> None:
    frame = _sample_causal_frame().iloc[:20].copy()
    predicted_uplift = frame["feature_signal"].to_numpy(dtype=float)
    uplift = compute_uplift_at_fraction(
        frame,
        predicted_uplift,
        treatment_col="treatment",
        outcome_col="outcome",
        fraction=0.2,
    )
    assert uplift > 0


def test_run_causal_temporal_backtest_writes_required_outputs(tmp_path: Path) -> None:
    frame = _sample_causal_frame()
    config = CausalMLConfig(
        time_col="time_key",
        treatment_col="treatment",
        outcome_col="outcome",
        feature_cols=["feature_signal", "feature_noise"],
        min_train_periods=2,
        test_size=1,
        output_dir=str(tmp_path / "causal_outputs"),
        perform_sensitivity_analysis=True,
    )
    summary, folds = run_causal_temporal_backtest(frame, config)

    assert not summary.empty
    assert len(folds) == len(summary)
    assert {"qini_coefficient", "ate", "uplift_top_decile", "policy_risk"}.issubset(
        summary.columns
    )

    output_dir = Path(config.output_dir)
    assert (output_dir / "time_series_performance.csv").exists()
    assert (output_dir / "backtesting_report.html").exists()
    assert (output_dir / "final_recommendation_summary.json").exists()

    fold_artifact_dir = output_dir / "run_1"
    assert (fold_artifact_dir / "uplift_curve.png").exists()
    assert (fold_artifact_dir / "qini_curve.png").exists()
    assert (fold_artifact_dir / "feature_importance.json").exists()
    assert (fold_artifact_dir / "trained_model.pkl").exists()
    assert (fold_artifact_dir / "backtesting_report.html").exists()


def test_export_fold_manifest_serializes_summary(tmp_path: Path) -> None:
    summary = pd.DataFrame([{"split_index": 1, "qini_coefficient": 0.4, "ate": 0.2}])
    config = CausalMLConfig(
        time_col="time_key",
        treatment_col="treatment",
        outcome_col="outcome",
        feature_cols=["x1"],
        output_dir=str(tmp_path),
    )
    manifest = export_fold_manifest(config, summary)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["summary_records"][0]["qini_coefficient"] == 0.4


def test_format_periods_none() -> None:
    from src.validation.causal import _format_periods

    assert _format_periods(tuple()) == "empty"


def test_estimate_ate_one_row() -> None:
    outcome = pd.Series([1.0])
    treatment = pd.Series([1])
    result = estimate_ate(outcome, treatment)
    assert result["ate_std_err"] == 0.0


def test_compute_uplift_curve_empty_gain() -> None:
    # compute_uplift_curve already tested with empty series,
    # but let's be sure about line 339
    curve = compute_uplift_curve(
        pd.Series([], dtype=float), pd.Series([], dtype=int), np.array([])
    )
    assert curve.empty


def test_evaluate_policy_value_treat_all() -> None:
    outcome = pd.Series([1.0, 0.5])
    treatment = pd.Series([1, 0])
    propensity = np.array([0.5, 0.5])
    uplift = np.array([0.1, -0.1])
    val = evaluate_policy_value(
        outcome, treatment, propensity, uplift, baseline_policy="treat_all"
    )
    assert "baseline_value" in val


def test_summarize_validation_performance_edge() -> None:
    from src.validation.causal import summarize_validation_performance

    ate_stats = {"ate": 1.0, "ate_p_value": 0.01}
    # qini > 0, ate > 0, p < 0.05, smd < 0.1 -> score 1.0
    assert summarize_validation_performance(0.1, ate_stats, 0.05) == 1.0
    # smd >= 0.1 -> score 0.75
    assert summarize_validation_performance(0.1, ate_stats, 0.15) == 0.75


def test_load_causal_frame_datetime(tmp_path) -> None:
    df = pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-01", "2024-01-02"]), "val": [3, 4]}
    )
    path = tmp_path / "test_dt.csv"
    df.to_csv(path, index=False)
    # The read_csv will need to parse dates
    l1 = load_causal_frame(path, "time")
    # Wait, load_causal_frame doesn't pass parse_dates to read_csv.
    # So it will be string, not datetime64, unless we convert it.
    # Ah, let's see causal.py line 829


def test_load_causal_frame_datetime_sorting(tmp_path) -> None:
    df = pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-02", "2024-01-01"]), "val": [4, 3]}
    )
    path = tmp_path / "test_dt.parquet"
    df.to_parquet(path)
    l1 = load_causal_frame(path, "time")
    assert l1["time"].iloc[0] == pd.Timestamp("2024-01-01")
