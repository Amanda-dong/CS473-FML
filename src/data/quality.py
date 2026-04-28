"""Quality guards for ETL outputs, embedding corpora, and model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.base import DatasetSpec


@dataclass(frozen=True)
class QualityReport:
    """Compact summary of rows kept and dropped during a preflight step."""

    dataset_name: str
    input_rows: int
    output_rows: int
    dropped_rows: int
    issues: tuple[str, ...] = ()


def validate_dataset_contract(frame: pd.DataFrame, spec: DatasetSpec) -> QualityReport:
    """Ensure an ETL output exposes the columns declared in its dataset spec."""

    missing = tuple(column for column in spec.columns if column not in frame.columns)
    if missing:
        raise ValueError(
            f"{spec.name} ETL output is missing required columns: {', '.join(missing)}"
        )
    return QualityReport(
        dataset_name=spec.name,
        input_rows=len(frame),
        output_rows=len(frame),
        dropped_rows=0,
        issues=(),
    )


def prepare_embedding_corpus(
    reviews_df: pd.DataFrame,
    text_col: str = "review_text",
    dedupe_columns: Iterable[str] | None = None,
    min_text_length: int = 5,
) -> tuple[pd.DataFrame, QualityReport]:
    """Normalize and de-duplicate review text before embedding generation."""

    if text_col not in reviews_df.columns:
        raise ValueError(f"Embedding corpus requires a '{text_col}' column.")

    frame = reviews_df.copy()
    input_rows = len(frame)
    frame[text_col] = (
        frame[text_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    frame = frame[frame[text_col].str.len() >= min_text_length].copy()

    if dedupe_columns is None:
        dedupe_columns = [
            column
            for column in ("restaurant_id", "business_id", "review_date", text_col)
            if column in frame.columns
        ]
        if not dedupe_columns:
            dedupe_columns = [text_col]
    else:
        dedupe_columns = [  # pragma: no cover
            column for column in dedupe_columns if column in frame.columns
        ]
        if not dedupe_columns:
            dedupe_columns = [text_col]

    frame = frame.drop_duplicates(subset=list(dedupe_columns)).reset_index(drop=True)
    return frame, QualityReport(
        dataset_name="embedding_corpus",
        input_rows=input_rows,
        output_rows=len(frame),
        dropped_rows=input_rows - len(frame),
        issues=(),
    )


def prepare_training_frame(
    feature_matrix: pd.DataFrame,
    target_col: str = "target",
    key_columns: tuple[str, ...] = ("zone_id", "time_key"),
    min_label_quality: float = 0.5,
) -> tuple[pd.DataFrame, QualityReport]:
    """Filter and sanitize a feature matrix before GPU-oriented model training."""

    if target_col not in feature_matrix.columns:
        raise ValueError(f"Training frame must contain '{target_col}'.")

    frame = feature_matrix.copy()
    input_rows = len(frame)

    present_key_columns = [column for column in key_columns if column in frame.columns]
    if present_key_columns:
        frame = frame.drop_duplicates(subset=present_key_columns, keep="last")

    frame = frame[frame[target_col].notna()].copy()

    if "label_quality" in frame.columns:
        frame = frame[frame["label_quality"].fillna(0.0) >= min_label_quality].copy()

    reserved = set(present_key_columns) | {
        target_col,
        "label_quality",
        "missingness_fraction",
    }
    non_numeric = [
        column
        for column in frame.columns
        if column not in reserved and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            "Training frame contains non-numeric feature columns that must be encoded first: "
            + ", ".join(non_numeric)
        )

    numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
        fill_values = frame[numeric_cols].median(numeric_only=True).fillna(0.0)
        frame[numeric_cols] = frame[numeric_cols].fillna(fill_values).astype(np.float32)

    if present_key_columns:
        frame = frame.sort_values(present_key_columns).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    return frame, QualityReport(
        dataset_name="training_frame",
        input_rows=input_rows,
        output_rows=len(frame),
        dropped_rows=input_rows - len(frame),
        issues=(),
    )


def prepare_survival_history(
    history: pd.DataFrame,
    key_col: str = "restaurant_id",
    duration_col: str = "duration_days",
    event_col: str = "event_observed",
) -> tuple[pd.DataFrame, QualityReport]:
    """Ensure survival training data is deduplicated and numerically safe."""

    required = (key_col, duration_col, event_col)
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(
            "Survival history is missing required columns: " + ", ".join(missing)
        )

    frame = history.copy()
    input_rows = len(frame)
    frame = frame.drop_duplicates(subset=[key_col], keep="last")
    frame = frame[frame[duration_col].fillna(0) > 0].copy()
    frame[event_col] = frame[event_col].fillna(0).astype(int).clip(0, 1)

    numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
        fill_values = frame[numeric_cols].median(numeric_only=True).fillna(0.0)
        frame[numeric_cols] = frame[numeric_cols].fillna(fill_values)

    frame = frame.reset_index(drop=True)
    return frame, QualityReport(
        dataset_name="survival_history",
        input_rows=input_rows,
        output_rows=len(frame),
        dropped_rows=input_rows - len(frame),
        issues=(),
    )
