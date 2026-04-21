"""Temporal validation helpers for the ML workstream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """A blocked train/test split over ordered time periods."""

    train_periods: tuple[object, ...]
    test_periods: tuple[object, ...]


def train_test_split_by_cutoff(
    frame: pd.DataFrame,
    time_col: str,
    train_end,
    test_start,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train and test windows using explicit cutoffs."""

    train_frame = frame.loc[frame[time_col] <= train_end].copy()
    test_frame = frame.loc[frame[time_col] >= test_start].copy()
    return train_frame, test_frame


def build_blocked_splits(
    frame: pd.DataFrame,
    time_col: str,
    min_train_periods: int = 3,
    test_size: int = 1,
) -> list[TemporalSplit]:
    """Build rolling blocked splits from sorted unique time periods."""

    periods = sorted(pd.Series(frame[time_col]).dropna().unique().tolist())
    splits: list[TemporalSplit] = []
    for split_end in range(min_train_periods, len(periods) - test_size + 1):
        train_periods = tuple(periods[:split_end])
        test_periods = tuple(periods[split_end : split_end + test_size])
        splits.append(
            TemporalSplit(train_periods=train_periods, test_periods=test_periods)
        )
    return splits


def apply_temporal_split(
    frame: pd.DataFrame,
    time_col: str,
    split: TemporalSplit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a frame to the periods specified by a TemporalSplit."""

    train_frame = frame.loc[frame[time_col].isin(split.train_periods)].copy()
    test_frame = frame.loc[frame[time_col].isin(split.test_periods)].copy()
    return train_frame, test_frame


def evaluate_top_k(recommended: list[str], observed: list[str], k: int = 5) -> float:
    """Compute a simple recall-at-k style metric for case-study evaluation."""

    if k <= 0:
        raise ValueError("k must be positive.")
    top_k = recommended[:k]
    if not observed:
        return 0.0
    hits = len(set(top_k).intersection(observed))
    return hits / len(set(observed))


# ---------------------------------------------------------------------------
# Phase 6 -- evaluation metrics
# ---------------------------------------------------------------------------


def ndcg_at_k(
    predicted_scores: np.ndarray, true_relevance: np.ndarray, k: int
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses the standard formula:
        DCG@k  = sum_{i=0}^{k-1} true_relevance[order[i]] / log2(i + 2)
        NDCG@k = DCG@k / IDCG@k
    where order sorts by predicted_scores descending, and IDCG uses ideal
    (true_relevance sorted descending) ordering.
    """
    predicted_scores = np.asarray(predicted_scores, dtype=float)
    true_relevance = np.asarray(true_relevance, dtype=float)

    # Sort indices by predicted score descending
    order = np.argsort(-predicted_scores)[:k]
    # DCG: sum of true_relevance[order[i]] / log2(i + 2) for i in 0..k-1
    dcg = sum(
        true_relevance[order[i]] / np.log2(i + 2) for i in range(min(k, len(order)))
    )
    # IDCG: same with ideal ordering (sort true_relevance descending)
    ideal_order = np.argsort(-true_relevance)[:k]
    idcg = sum(
        true_relevance[ideal_order[i]] / np.log2(i + 2)
        for i in range(min(k, len(ideal_order)))
    )
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def expected_calibration_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: weighted average of |predicted - actual| per bin.

    Bins predictions into ``n_bins`` equal-width intervals over [0, 1] and
    computes the weighted-average absolute difference between the mean
    predicted value and the mean actual outcome in each bin, weighted by
    bin size.

    Parameters
    ----------
    predicted : array-like
        Predicted probabilities or scores in [0, 1].
    actual : array-like
        Binary outcomes (0 or 1) or continuous ground truth in [0, 1].
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        The ECE value (lower is better).
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n = len(predicted)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted == bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        avg_pred = float(np.mean(predicted[mask]))
        avg_actual = float(np.mean(actual[mask]))
        ece += (count / n) * abs(avg_pred - avg_actual)
    return ece


def bootstrap_metric(
    metric_fn,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for any metric.

    Parameters
    ----------
    metric_fn : callable
        Function with signature ``metric_fn(y_pred, y_true) -> float``.
    y_pred : array-like
        Predicted values.
    y_true : array-like
        True values.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: ``mean``, ``std``, ``ci_lower``, ``ci_upper``.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_pred)

    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(float(metric_fn(y_pred[idx], y_true[idx])))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def mean_average_precision(
    predicted_ranking: list[str], true_top_zones: set[str]
) -> float:
    """MAP: average precision across ranked positions."""
    if not true_top_zones:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, zone in enumerate(predicted_ranking, 1):
        if zone in true_top_zones:
            hits += 1
            precision_sum += hits / i
    if hits == 0:
        return 0.0
    return precision_sum / len(true_top_zones)


def calibration_analysis(
    predicted_scores: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bin predictions into quantiles, compute mean actual outcome per bin."""
    df = pd.DataFrame({"predicted": predicted_scores, "actual": actual_outcomes})
    df["bin"] = pd.qcut(df["predicted"], q=n_bins, duplicates="drop")
    result = (
        df.groupby("bin", observed=True)
        .agg(
            mean_predicted=("predicted", "mean"),
            mean_actual=("actual", "mean"),
            count=("actual", "size"),
        )
        .reset_index()
    )
    result.rename(columns={"bin": "bin_label"}, inplace=True)
    return result


def _precision_at_k(predicted_ranking: list[str], true_top: set[str], k: int) -> float:
    top_k = predicted_ranking[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & true_top) / k


def _resolve_target_column(ground_truth: pd.DataFrame, year_col: str) -> str:
    """Pick the intended training target from a ground-truth frame."""
    for candidate in ("y_composite", "target"):
        if candidate in ground_truth.columns:
            return candidate

    reserved = {"zone_id", year_col, "missingness_fraction", "label_quality"}
    candidates = [column for column in ground_truth.columns if column not in reserved]
    if not candidates:
        raise ValueError("Ground-truth frame does not contain a usable target column.")
    return candidates[-1]


def run_temporal_backtest(
    feature_matrix: pd.DataFrame,
    ground_truth: pd.DataFrame,
    model_cls,
    year_col: str = "time_key",
    min_train_years: int = 2,
) -> pd.DataFrame:
    """Walk-forward temporal backtest with expanding window.

    For each year Y from min_train_years onward:
      - Train on all years < Y (expanding window)
      - Predict on year Y
      - Compute NDCG@5, NDCG@10, precision@5, MAP, calibration error, ECE
      - Compute bootstrap 95% CI for NDCG@5

    Returns DataFrame with per-fold metrics and confidence intervals.
    """
    years = sorted(feature_matrix[year_col].unique())
    records: list[dict] = []

    target_col = _resolve_target_column(ground_truth, year_col)

    for idx in range(min_train_years, len(years)):
        test_year = years[idx]
        train_years = years[:idx]

        drop_cols = [c for c in [year_col, "zone_id"] if c in feature_matrix.columns]
        train_X = feature_matrix[feature_matrix[year_col].isin(train_years)].drop(
            columns=drop_cols
        )
        train_y = (
            ground_truth.loc[train_X.index, target_col]
            if target_col in ground_truth.columns
            else ground_truth.iloc[train_X.index.to_list(), -1]
        )

        test_X = feature_matrix[feature_matrix[year_col] == test_year].drop(
            columns=drop_cols
        )
        test_y = (
            ground_truth.loc[test_X.index, target_col]
            if target_col in ground_truth.columns
            else ground_truth.iloc[test_X.index.to_list(), -1]
        )

        if len(train_X) == 0 or len(test_X) == 0:
            continue

        model = model_cls()
        model.fit(train_X, train_y)
        preds = model.predict(test_X)

        pred_arr = np.asarray(preds, dtype=float)
        true_arr = np.asarray(test_y, dtype=float)

        n5 = ndcg_at_k(pred_arr, true_arr, 5)
        n10 = ndcg_at_k(pred_arr, true_arr, 10)

        # Bootstrap CI for NDCG@5
        n5_ci = bootstrap_metric(
            lambda p, t: ndcg_at_k(p, t, 5),
            pred_arr,
            true_arr,
            n_bootstrap=500,
            ci=0.95,
        )

        # For MAP and precision, define "true top" as top-5 by actual outcome
        true_order = np.argsort(-true_arr)
        true_top_ids = [str(i) for i in true_order[:5]]
        pred_order = np.argsort(-pred_arr)
        pred_ranking = [str(i) for i in pred_order]

        p5 = _precision_at_k(pred_ranking, set(true_top_ids), 5)
        map_s = mean_average_precision(pred_ranking, set(true_top_ids))

        # Calibration error: mean absolute diff between bin means
        if len(pred_arr) >= 5:
            cal = calibration_analysis(pred_arr, true_arr, n_bins=min(5, len(pred_arr)))
            cal_err = float((cal["mean_predicted"] - cal["mean_actual"]).abs().mean())
        else:
            cal_err = float(np.abs(pred_arr.mean() - true_arr.mean()))

        # ECE (normalize to [0,1] for ECE if not already)
        pred_norm = pred_arr.copy()
        true_norm = true_arr.copy()
        p_range = pred_norm.max() - pred_norm.min()
        t_range = true_norm.max() - true_norm.min()
        if p_range > 0:
            pred_norm = (pred_norm - pred_norm.min()) / p_range
        if t_range > 0:
            true_norm = (true_norm - true_norm.min()) / t_range
        ece = expected_calibration_error(
            pred_norm, true_norm, n_bins=min(10, len(pred_arr))
        )

        records.append(
            {
                "year": test_year,
                "n_train": len(train_X),
                "n_test": len(test_X),
                "ndcg_5": n5,
                "ndcg_5_ci_lower": n5_ci["ci_lower"],
                "ndcg_5_ci_upper": n5_ci["ci_upper"],
                "ndcg_10": n10,
                "precision_5": p5,
                "map_score": map_s,
                "cal_error": cal_err,
                "ece": ece,
            }
        )

    return pd.DataFrame(records)
