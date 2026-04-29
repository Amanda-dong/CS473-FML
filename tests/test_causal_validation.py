"""Tests for causal uplift evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.causal import (
    CausalMLConfig,
    compute_qini_coefficient,
    compute_standardized_mean_differences,
    compute_uplift_at_fraction,
    compute_uplift_curve,
    estimate_ate,
    estimate_propensity_scores,
    export_fold_manifest,
    make_temporal_splits,
    run_causal_temporal_backtest,
)


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
    assert {"qini_coefficient", "ate", "uplift_top_decile", "policy_risk"}.issubset(summary.columns)

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
