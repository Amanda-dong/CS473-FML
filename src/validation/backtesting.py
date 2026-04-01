"""Temporal validation helpers for the ML workstream."""

from __future__ import annotations

from dataclasses import dataclass

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
        splits.append(TemporalSplit(train_periods=train_periods, test_periods=test_periods))
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
