"""Tests for temporal validation helpers."""

import pandas as pd

from src.validation.backtesting import apply_temporal_split, build_blocked_splits, evaluate_top_k


def test_build_blocked_splits_returns_expected_count() -> None:
    """Rolling splits should be created from sorted time periods."""

    frame = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022]})
    splits = build_blocked_splits(frame, time_col="year", min_train_periods=3, test_size=1)
    assert len(splits) == 2


def test_apply_temporal_split_filters_periods() -> None:
    """The helper should return only rows from the requested windows."""

    frame = pd.DataFrame({"year": [2019, 2020, 2021], "value": [1, 2, 3]})
    split = build_blocked_splits(frame, time_col="year", min_train_periods=2, test_size=1)[0]
    train_frame, test_frame = apply_temporal_split(frame, "year", split)
    assert train_frame["year"].tolist() == [2019, 2020]
    assert test_frame["year"].tolist() == [2021]


def test_evaluate_top_k_is_bounded() -> None:
    """Recall-at-k helper should return values between zero and one."""

    score = evaluate_top_k(["a", "b", "c"], ["b", "d"], k=2)
    assert 0.0 <= score <= 1.0
