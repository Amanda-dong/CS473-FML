"""Causal uplift evaluation helpers with temporal backtesting and MLflow logging."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from src.validation.backtesting import TemporalSplit, apply_temporal_split


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.asarray(denominator, dtype=float) != 0,
    )


def _as_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.select_dtypes(include="number")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _format_periods(periods: tuple[object, ...]) -> str:
    if not periods:
        return "empty"
    if len(periods) == 1:
        return str(periods[0])
    return f"{periods[0]} to {periods[-1]}"


@dataclass(frozen=True)
class CausalMLConfig:
    """Configuration for a causal uplift evaluation run."""

    time_col: str
    treatment_col: str
    outcome_col: str
    feature_cols: list[str]
    propensity_col: str | None = None
    model_type: str = "t_learner_gbr"
    feature_set_version: str = "v1"
    treatment_definition: str = "model_driven_action"
    outcome_definition: str = "outcome"
    min_train_periods: int = 3
    test_size: int = 1
    window_type: str = "expanding"
    top_fraction: float = 0.1
    random_state: int = 42
    experiment_name: str = "causal_uplift_model_v1"
    output_dir: str = "data/processed/causal_uplift_model_v1"
    mlflow_tracking_uri: str | None = None
    baseline_policy: str = "no_treatment"
    perform_sensitivity_analysis: bool = True


@dataclass(frozen=True)
class TemporalBacktestFold:
    split_index: int
    split: TemporalSplit
    metrics: dict[str, float]
    train_rows: int
    test_rows: int
    artifacts: dict[str, str]
    recommendation: str


class TLearnerUpliftModel:
    """Simple T-learner for estimating conditional treatment effects."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.treated_model = GradientBoostingRegressor(random_state=random_state)
        self.control_model = GradientBoostingRegressor(random_state=random_state)
        self.feature_names_: list[str] = []
        self.feature_importance_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> None:
        X_numeric = _as_numeric_frame(X)
        treatment = pd.Series(treatment).astype(int)
        outcome = pd.Series(outcome).astype(float)
        self.feature_names_ = list(X_numeric.columns)

        treated_mask = treatment == 1
        control_mask = treatment == 0
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("Both treated and control samples are required.")

        self.treated_model.fit(X_numeric.loc[treated_mask], outcome.loc[treated_mask])
        self.control_model.fit(X_numeric.loc[control_mask], outcome.loc[control_mask])

        treated_importance = getattr(
            self.treated_model, "feature_importances_", np.zeros(len(self.feature_names_))
        )
        control_importance = getattr(
            self.control_model, "feature_importances_", np.zeros(len(self.feature_names_))
        )
        avg_importance = (treated_importance + control_importance) / 2.0
        self.feature_importance_ = {
            name: float(value)
            for name, value in zip(self.feature_names_, avg_importance, strict=False)
        }

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = _as_numeric_frame(X).reindex(columns=self.feature_names_, fill_value=0.0)
        mu1 = self.treated_model.predict(X_numeric)
        mu0 = self.control_model.predict(X_numeric)
        return np.asarray(mu1 - mu0, dtype=float)

    def predict_potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_numeric = _as_numeric_frame(X).reindex(columns=self.feature_names_, fill_value=0.0)
        mu1 = np.asarray(self.treated_model.predict(X_numeric), dtype=float)
        mu0 = np.asarray(self.control_model.predict(X_numeric), dtype=float)
        return mu1, mu0


def make_temporal_splits(
    frame: pd.DataFrame,
    time_col: str,
    min_train_periods: int = 3,
    test_size: int = 1,
    window_type: str = "expanding",
) -> list[TemporalSplit]:
    """Create expanding or rolling temporal windows without leakage."""

    periods = sorted(pd.Series(frame[time_col]).dropna().unique().tolist())
    splits: list[TemporalSplit] = []
    for split_end in range(min_train_periods, len(periods) - test_size + 1):
        if window_type == "rolling":
            train_periods = tuple(periods[split_end - min_train_periods : split_end])
        else:
            train_periods = tuple(periods[:split_end])
        test_periods = tuple(periods[split_end : split_end + test_size])
        splits.append(
            TemporalSplit(train_periods=train_periods, test_periods=test_periods)
        )
    return splits


def estimate_propensity_scores(
    X: pd.DataFrame,
    treatment: pd.Series,
    model: Any | None = None,
    clip_min: float = 0.05,
    clip_max: float = 0.95,
) -> np.ndarray:
    """Estimate treatment propensity scores and clip extreme values."""

    X_numeric = _as_numeric_frame(X)
    treatment = pd.Series(treatment).astype(int)
    propensity_model = clone(model) if model is not None else LogisticRegression(max_iter=1000)
    propensity_model.fit(X_numeric, treatment)
    if hasattr(propensity_model, "predict_proba"):
        propensity = propensity_model.predict_proba(X_numeric)[:, 1]
    else:
        decision = np.asarray(propensity_model.decision_function(X_numeric), dtype=float)
        propensity = 1.0 / (1.0 + np.exp(-decision))
    return np.clip(propensity, clip_min, clip_max)


def compute_standardized_mean_differences(
    frame: pd.DataFrame,
    treatment_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute covariate balance diagnostics via standardized mean differences."""

    numeric = _as_numeric_frame(frame[feature_cols])
    treatment = frame[treatment_col].astype(int)
    treated = numeric.loc[treatment == 1]
    control = numeric.loc[treatment == 0]
    rows: list[dict[str, float | str]] = []
    for column in numeric.columns:
        treated_values = treated[column].to_numpy(dtype=float)
        control_values = control[column].to_numpy(dtype=float)
        treated_mean = _safe_mean(treated_values)
        control_mean = _safe_mean(control_values)
        treated_var = float(np.var(treated_values, ddof=1)) if len(treated_values) > 1 else 0.0
        control_var = float(np.var(control_values, ddof=1)) if len(control_values) > 1 else 0.0
        pooled_sd = math.sqrt(max((treated_var + control_var) / 2.0, 0.0))
        if pooled_sd == 0:
            smd = 0.0 if treated_mean == control_mean else math.copysign(float("inf"), treated_mean - control_mean)
        else:
            smd = (treated_mean - control_mean) / pooled_sd
        rows.append(
            {
                "feature": column,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def estimate_ate(
    outcome: pd.Series,
    treatment: pd.Series,
    propensity: np.ndarray | None = None,
) -> dict[str, float]:
    """Estimate ATE from observed data with naive and IPW estimators."""

    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    naive_ate = _safe_mean(y[t == 1]) - _safe_mean(y[t == 0])

    if propensity is None:
        propensity = np.full_like(y, fill_value=np.clip(t.mean(), 0.05, 0.95), dtype=float)
    propensity = np.clip(np.asarray(propensity, dtype=float), 0.05, 0.95)

    influence = (t * y / propensity) - ((1 - t) * y / (1 - propensity))
    ipw_ate = float(np.mean(influence))
    se = float(np.std(influence, ddof=1) / math.sqrt(len(influence))) if len(influence) > 1 else 0.0
    z_score = 0.0 if se == 0 else ipw_ate / se
    p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0)))
    ci_delta = 1.96 * se
    return {
        "ate": ipw_ate,
        "ate_naive": float(naive_ate),
        "ate_std_err": se,
        "ate_ci_lower": float(ipw_ate - ci_delta),
        "ate_ci_upper": float(ipw_ate + ci_delta),
        "ate_p_value": p_value,
        "ate_z_score": float(z_score),
    }


def compute_uplift_curve(
    outcome: pd.Series,
    treatment: pd.Series,
    predicted_uplift: np.ndarray,
) -> pd.DataFrame:
    """Compute cumulative incremental gain after ranking by predicted uplift."""

    df = pd.DataFrame(
        {
            "outcome": np.asarray(outcome, dtype=float),
            "treatment": np.asarray(treatment, dtype=int),
            "predicted_uplift": np.asarray(predicted_uplift, dtype=float),
        }
    ).sort_values("predicted_uplift", ascending=False, kind="mergesort")
    df["rank"] = np.arange(1, len(df) + 1)
    df["population_fraction"] = df["rank"] / max(len(df), 1)
    df["cum_treated"] = df["treatment"].cumsum()
    df["cum_control"] = (1 - df["treatment"]).cumsum()
    df["cum_treated_outcome"] = (df["outcome"] * df["treatment"]).cumsum()
    df["cum_control_outcome"] = (df["outcome"] * (1 - df["treatment"])).cumsum()
    ratio = _safe_divide(
        df["cum_treated"].to_numpy(dtype=float),
        df["cum_control"].to_numpy(dtype=float),
    )
    df["incremental_gain"] = (
        df["cum_treated_outcome"]
        - df["cum_control_outcome"] * ratio
    )
    total_gain = float(df["incremental_gain"].iloc[-1]) if not df.empty else 0.0
    df["random_baseline"] = df["population_fraction"] * total_gain
    df["uplift_curve"] = df["incremental_gain"]
    return df.reset_index(drop=True)


def compute_qini_coefficient(uplift_curve: pd.DataFrame) -> float:
    """Area between the model uplift curve and a random baseline."""

    if uplift_curve.empty:
        return 0.0
    x = uplift_curve["population_fraction"].to_numpy(dtype=float)
    y_model = uplift_curve["uplift_curve"].to_numpy(dtype=float)
    y_random = uplift_curve["random_baseline"].to_numpy(dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_model - y_random, x=x))
    return float(np.trapz(y_model - y_random, x=x))


def compute_uplift_at_fraction(
    frame: pd.DataFrame,
    predicted_uplift: np.ndarray,
    treatment_col: str,
    outcome_col: str,
    fraction: float = 0.1,
) -> float:
    """Observed uplift in the highest-ranked fraction of the population."""

    if frame.empty:
        return 0.0
    top_n = max(1, int(math.ceil(len(frame) * fraction)))
    ranked = frame.assign(predicted_uplift=np.asarray(predicted_uplift, dtype=float)).nlargest(
        top_n, "predicted_uplift"
    )
    treated_mean = _safe_mean(ranked.loc[ranked[treatment_col] == 1, outcome_col])
    control_mean = _safe_mean(ranked.loc[ranked[treatment_col] == 0, outcome_col])
    return float(treated_mean - control_mean)


def evaluate_policy_value(
    outcome: pd.Series,
    treatment: pd.Series,
    propensity: np.ndarray,
    predicted_uplift: np.ndarray,
    baseline_policy: str = "no_treatment",
) -> dict[str, float]:
    """Estimate policy value under an uplift-based treatment rule."""

    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    p = np.clip(np.asarray(propensity, dtype=float), 0.05, 0.95)
    policy = (np.asarray(predicted_uplift, dtype=float) > 0).astype(int)

    policy_value = float(
        np.mean(
            (policy * t * y / p) + ((1 - policy) * (1 - t) * y / (1 - p))
        )
    )
    if baseline_policy == "treat_all":
        baseline = float(np.mean(t * y / p))
    elif baseline_policy == "historical":
        baseline = float(np.mean(y))
    else:
        baseline = float(np.mean((1 - t) * y / (1 - p)))
    return {
        "policy_value": policy_value,
        "baseline_value": baseline,
        "policy_risk": float(baseline - policy_value),
        "policy_treated_rate": float(policy.mean()),
    }


def summarize_validation_performance(
    qini_coefficient: float,
    ate_stats: dict[str, float],
    max_abs_smd: float,
) -> float:
    """Compact validation score where larger is better."""

    score = 0.0
    score += 1.0 if qini_coefficient > 0 else 0.0
    score += 1.0 if ate_stats["ate"] > 0 else 0.0
    score += 1.0 if ate_stats["ate_p_value"] < 0.05 else 0.0
    score += 1.0 if max_abs_smd < 0.1 else 0.0
    return score / 4.0


def run_sensitivity_analysis(
    frame: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    propensity: np.ndarray,
    random_state: int = 42,
) -> pd.DataFrame:
    """Perturb treatment assignments to stress-test the ATE estimate."""

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []
    for noise_level in (0.0, 0.05, 0.1, 0.15, 0.2):
        perturbed = frame[treatment_col].astype(int).to_numpy(copy=True)
        flips = rng.random(len(perturbed)) < noise_level
        perturbed[flips] = 1 - perturbed[flips]
        ate_stats = estimate_ate(frame[outcome_col], perturbed, propensity=propensity)
        rows.append({"noise_level": noise_level, "ate": ate_stats["ate"]})
    return pd.DataFrame(rows)


def _create_uplift_model(config: CausalMLConfig) -> TLearnerUpliftModel:
    if config.model_type == "t_learner_gbr":
        return TLearnerUpliftModel(random_state=config.random_state)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def evaluate_causal_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: CausalMLConfig,
) -> tuple[TLearnerUpliftModel, dict[str, float], dict[str, Any]]:
    """Fit an uplift model on a temporal train split and evaluate on test."""

    model = _create_uplift_model(config)
    model.fit(
        train_frame[config.feature_cols],
        train_frame[config.treatment_col],
        train_frame[config.outcome_col],
    )

    if config.propensity_col and config.propensity_col in test_frame.columns:
        propensity = np.clip(
            test_frame[config.propensity_col].to_numpy(dtype=float),
            0.05,
            0.95,
        )
    else:
        propensity_model = GradientBoostingClassifier(random_state=config.random_state)
        propensity_model.fit(
            _as_numeric_frame(train_frame[config.feature_cols]),
            train_frame[config.treatment_col].astype(int),
        )
        propensity = np.clip(
            propensity_model.predict_proba(_as_numeric_frame(test_frame[config.feature_cols]))[:, 1],
            0.05,
            0.95,
        )

    predicted_uplift = model.predict_uplift(test_frame[config.feature_cols])
    mu1, mu0 = model.predict_potential_outcomes(test_frame[config.feature_cols])
    uplift_curve = compute_uplift_curve(
        outcome=test_frame[config.outcome_col],
        treatment=test_frame[config.treatment_col],
        predicted_uplift=predicted_uplift,
    )
    qini = compute_qini_coefficient(uplift_curve)
    ate_stats = estimate_ate(
        outcome=test_frame[config.outcome_col],
        treatment=test_frame[config.treatment_col],
        propensity=propensity,
    )
    uplift_top_decile = compute_uplift_at_fraction(
        test_frame,
        predicted_uplift,
        treatment_col=config.treatment_col,
        outcome_col=config.outcome_col,
        fraction=config.top_fraction,
    )
    policy_stats = evaluate_policy_value(
        outcome=test_frame[config.outcome_col],
        treatment=test_frame[config.treatment_col],
        propensity=propensity,
        predicted_uplift=predicted_uplift,
        baseline_policy=config.baseline_policy,
    )
    balance = compute_standardized_mean_differences(
        train_frame,
        treatment_col=config.treatment_col,
        feature_cols=config.feature_cols,
    )
    max_abs_smd = float(balance["abs_smd"].max()) if not balance.empty else 0.0
    validation_performance = summarize_validation_performance(
        qini_coefficient=qini,
        ate_stats=ate_stats,
        max_abs_smd=max_abs_smd,
    )

    metrics = {
        "qini_coefficient": qini,
        "ate": ate_stats["ate"],
        "ate_naive": ate_stats["ate_naive"],
        "ate_p_value": ate_stats["ate_p_value"],
        "ate_ci_lower": ate_stats["ate_ci_lower"],
        "ate_ci_upper": ate_stats["ate_ci_upper"],
        "uplift_top_decile": uplift_top_decile,
        "policy_risk": policy_stats["policy_risk"],
        "policy_value": policy_stats["policy_value"],
        "baseline_value": policy_stats["baseline_value"],
        "validation_performance": validation_performance,
        "max_abs_smd": max_abs_smd,
        "mean_predicted_uplift": float(np.mean(predicted_uplift)),
        "positive_uplift_share": float(np.mean(predicted_uplift > 0)),
    }
    artifacts = {
        "uplift_curve": uplift_curve,
        "balance_table": balance,
        "sensitivity_analysis": (
            run_sensitivity_analysis(
                test_frame,
                treatment_col=config.treatment_col,
                outcome_col=config.outcome_col,
                propensity=propensity,
                random_state=config.random_state,
            )
            if config.perform_sensitivity_analysis
            else pd.DataFrame()
        ),
        "feature_importance": model.feature_importance_,
        "predictions": test_frame.assign(
            predicted_uplift=predicted_uplift,
            predicted_y1=mu1,
            predicted_y0=mu0,
            propensity=propensity,
        ),
    }
    return model, metrics, artifacts


def _plot_curve(
    curve_frame: pd.DataFrame,
    y_col: str,
    baseline_col: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(curve_frame["population_fraction"], curve_frame[y_col], label="Model")
    plt.plot(
        curve_frame["population_fraction"],
        curve_frame[baseline_col],
        linestyle="--",
        label="Random baseline",
    )
    plt.xlabel("Population Fraction")
    plt.ylabel("Incremental Outcome")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _write_html_report(
    summary: pd.DataFrame,
    recommendations: list[str],
    output_path: Path,
) -> None:
    table_html = summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    recommendation_html = "".join(f"<li>{item}</li>" for item in recommendations)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Causal Uplift Backtesting Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Causal Uplift Backtesting Summary</h1>
  {table_html}
  <h2>Final Recommendation Summary</h2>
  <ul>{recommendation_html}</ul>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _maybe_start_mlflow_run(
    config: CausalMLConfig,
    run_name: str,
    tags: dict[str, str],
):
    try:
        import mlflow
    except ImportError:
        return None, None

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(tags)
    mlflow.log_params(
        {
            "model_type": config.model_type,
            "feature_set_version": config.feature_set_version,
            "training_window": config.window_type,
            "treatment_definition": config.treatment_definition,
            "outcome_definition": config.outcome_definition,
        }
    )
    return mlflow, run


def _log_mlflow_artifacts(
    mlflow_module: Any,
    metrics: dict[str, float],
    artifact_paths: dict[str, Path],
) -> None:
    mlflow_module.log_metrics(
        {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.floating))
        }
    )
    for path in artifact_paths.values():
        mlflow_module.log_artifact(str(path))


def _recommendation_from_metrics(metrics: dict[str, float]) -> str:
    if (
        metrics["qini_coefficient"] > 0
        and metrics["ate"] > 0
        and metrics["ate_p_value"] < 0.05
        and metrics["max_abs_smd"] < 0.1
    ):
        return "Pass: uplift is positive, statistically supported, and covariate balance is acceptable."
    if metrics["qini_coefficient"] <= 0:
        return "Fail: uplift does not beat the random baseline."
    if metrics["ate"] <= 0:
        return "Fail: estimated average treatment effect is not positive."
    if metrics["max_abs_smd"] >= 0.1:
        return "Caution: covariate imbalance is above the preferred threshold."
    return "Caution: results are mixed and should not be promoted yet."


def run_causal_temporal_backtest(
    frame: pd.DataFrame,
    config: CausalMLConfig,
) -> tuple[pd.DataFrame, list[TemporalBacktestFold]]:
    """Run temporal backtesting, write artifacts, and log each fold to MLflow."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = make_temporal_splits(
        frame,
        time_col=config.time_col,
        min_train_periods=config.min_train_periods,
        test_size=config.test_size,
        window_type=config.window_type,
    )

    summary_rows: list[dict[str, Any]] = []
    folds: list[TemporalBacktestFold] = []
    recommendations: list[str] = []

    for split_index, split in enumerate(splits, start=1):
        train_frame, test_frame = apply_temporal_split(frame, config.time_col, split)
        model, metrics, artifacts = evaluate_causal_split(train_frame, test_frame, config)
        recommendation = _recommendation_from_metrics(metrics)
        recommendations.append(
            f"Fold {split_index} ({_format_periods(split.test_periods)}): {recommendation}"
        )

        fold_dir = output_dir / f"run_{split_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        uplift_curve_path = fold_dir / "uplift_curve.png"
        qini_curve_path = fold_dir / "qini_curve.png"
        feature_importance_path = fold_dir / "feature_importance.json"
        trained_model_path = fold_dir / "trained_model.pkl"
        backtesting_report_path = fold_dir / "backtesting_report.html"
        performance_table_path = fold_dir / "time_series_performance.csv"

        _plot_curve(
            artifacts["uplift_curve"],
            y_col="uplift_curve",
            baseline_col="random_baseline",
            title="Uplift Curve",
            output_path=uplift_curve_path,
        )
        _plot_curve(
            artifacts["uplift_curve"],
            y_col="uplift_curve",
            baseline_col="random_baseline",
            title="Qini Curve",
            output_path=qini_curve_path,
        )

        feature_importance_path.write_text(
            json.dumps(artifacts["feature_importance"], indent=2),
            encoding="utf-8",
        )
        joblib.dump(model, trained_model_path)

        fold_summary = pd.DataFrame(
            [
                {
                    "split_index": split_index,
                    "train_period": _format_periods(split.train_periods),
                    "test_period": _format_periods(split.test_periods),
                    **metrics,
                }
            ]
        )
        fold_summary.to_csv(performance_table_path, index=False)
        _write_html_report(
            summary=fold_summary,
            recommendations=[recommendation],
            output_path=backtesting_report_path,
        )

        mlflow_module, _run = _maybe_start_mlflow_run(
            config,
            run_name=f"run_{split_index}",
            tags={
                "train_period": _format_periods(split.train_periods),
                "test_period": _format_periods(split.test_periods),
            },
        )
        if mlflow_module is not None:
            _log_mlflow_artifacts(
                mlflow_module,
                metrics,
                {
                    "uplift_curve": uplift_curve_path,
                    "qini_curve": qini_curve_path,
                    "feature_importance": feature_importance_path,
                    "trained_model": trained_model_path,
                    "backtesting_report": backtesting_report_path,
                },
            )
            mlflow_module.end_run()

        row = {
            "split_index": split_index,
            "train_period": _format_periods(split.train_periods),
            "test_period": _format_periods(split.test_periods),
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            **metrics,
        }
        summary_rows.append(row)
        folds.append(
            TemporalBacktestFold(
                split_index=split_index,
                split=split,
                metrics=metrics,
                train_rows=len(train_frame),
                test_rows=len(test_frame),
                artifacts={
                    "uplift_curve.png": str(uplift_curve_path),
                    "qini_curve.png": str(qini_curve_path),
                    "feature_importance.json": str(feature_importance_path),
                    "trained_model.pkl": str(trained_model_path),
                    "backtesting_report.html": str(backtesting_report_path),
                },
                recommendation=recommendation,
            )
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary_path = output_dir / "time_series_performance.csv"
        summary.to_csv(summary_path, index=False)

        stable_uplift = int((summary["qini_coefficient"] > 0).sum()) >= 3
        no_recent_degradation = bool(summary["qini_coefficient"].iloc[-1] > 0)
        passes_all = bool(
            stable_uplift
            and no_recent_degradation
            and (summary["ate"] > 0).all()
            and (summary["ate_p_value"] < 0.05).all()
            and (summary["max_abs_smd"] < 0.1).all()
        )
        registry_decision = {
            "production_ready": passes_all,
            "stable_uplift_splits": int((summary["qini_coefficient"] > 0).sum()),
            "latest_split_qini": float(summary["qini_coefficient"].iloc[-1]),
            "recommendation": (
                "Promote to Production"
                if passes_all
                else "Do not promote to Production"
            ),
        }
        registry_path = output_dir / "final_recommendation_summary.json"
        registry_path.write_text(json.dumps(registry_decision, indent=2), encoding="utf-8")

        report_path = output_dir / "backtesting_report.html"
        recommendation_lines = recommendations + [registry_decision["recommendation"]]
        _write_html_report(summary=summary, recommendations=recommendation_lines, output_path=report_path)
    return summary, folds


def load_causal_frame(
    dataset_path: str | Path,
    time_col: str,
) -> pd.DataFrame:
    """Load a causal evaluation frame from parquet or CSV."""

    path = Path(dataset_path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Supported dataset formats are .csv and .parquet only.")

    if time_col in frame.columns and np.issubdtype(frame[time_col].dtype, np.datetime64):
        frame = frame.sort_values(time_col).reset_index(drop=True)
    else:
        frame = frame.sort_values(time_col).reset_index(drop=True)
    return frame


def export_fold_manifest(
    config: CausalMLConfig,
    summary: pd.DataFrame,
) -> Path:
    """Write a compact machine-readable summary of the full run."""

    output_path = Path(config.output_dir) / "manifest.json"
    payload = {
        "config": asdict(config),
        "summary_records": summary.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path
