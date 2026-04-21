"""Tests for temporal validation helpers."""

import numpy as np
import pandas as pd

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
