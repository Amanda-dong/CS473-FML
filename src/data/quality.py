"""Quality guards for ETL outputs, embedding corpora, and model training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from src.config.constants import MODEL_CONFIG

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


def fill_feature_matrix_nulls(
    fm: pd.DataFrame, acs_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Impute missing values in the feature matrix using domain-specific logic."""
    df = fm.copy()
    if df.empty:
        return df

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # 1. Drop null zone_id
        df = df.dropna(subset=["zone_id"])

        # 2. Trip / Station counts
        for col in ["trip_count", "station_count", "permit_velocity"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 3. NLP share cols
        share_cols = [
            "halal_related_share",
            "explicit_halal_share",
            "implicit_halal_share",
            "not_related_share",
            "healthy_food_share",
            "salad_bowls_share",
            "mediterranean_bowls_share",
            "healthy_indian_share",
            "smoothie_juice_share",
            "halal_fast_casual_share",
            "subtype_gap",
        ]
        for col in share_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 4. NLP rate cols
        rate_cols = [
            "overall_positive_rate",
            "overall_negative_rate",
            "halal_positive_rate",
            "halal_negative_rate",
            "non_halal_positive_rate",
            "non_halal_negative_rate",
            "avg_rating",
            "avg_confidence",
        ]
        for col in rate_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.5)

        # 5. NLP count cols
        count_cols = [
            "total_review_count",
            "unique_restaurant_count",
            "halal_related_review_count",
            "explicit_halal_review_count",
            "implicit_halal_review_count",
            "not_related_review_count",
        ]
        for col in count_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 6. Dominant subtype
        if "dominant_subtype" in df.columns:
            df["dominant_subtype"] = df["dominant_subtype"].fillna("unknown")

        # 7. License velocity / net opens / closes / label quality
        for col in ["license_velocity", "net_opens", "net_closes", "label_quality"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 8. Target forward-fill and median fill
        if "target" in df.columns:
            df["target"] = df.groupby("zone_id")["target"].transform(
                lambda x: x.fillna(x.mean())
            )
            df["target"] = df["target"].fillna(df["target"].median())

        # 9. Inspection grade avg static
        if "inspection_grade_avg_static" in df.columns:
            df = df.sort_values(["zone_id", "time_key"])
            df["inspection_grade_avg_static"] = (
                df.groupby("zone_id")["inspection_grade_avg_static"].ffill().bfill()
            )
            df["inspection_grade_avg_static"] = df[
                "inspection_grade_avg_static"
            ].fillna(df["inspection_grade_avg_static"].median())

        # 10. ACS Fills
        if acs_df is not None:
            # Zone-id format: 'bk-tandon' or 'nta-bk0101'
            def get_borough(zid):
                if zid.startswith("nta-"):
                    code = zid[4:6].upper()
                else:
                    code = zid[:2].upper()
                _BOROUGH_CODES = {"BK", "MN", "QN", "BX", "SI"}
                return code if code in _BOROUGH_CODES else "UNKNOWN"

            df["borough"] = df["zone_id"].apply(get_borough)
            acs = acs_df
            acs = acs.assign(borough=acs["nta_id"].str[:2].str.upper())
            acs_borough_medians = (
                acs.groupby(["borough"])[["median_income", "population", "rent_burden"]]
                .median()
                .reset_index()
            )

            df = df.merge(
                acs_borough_medians, on=["borough"], how="left", suffixes=("", "_med")
            )
            for _acs_col in ("median_income", "population", "rent_burden"):
                _med_col = f"{_acs_col}_med"
                if _acs_col in df.columns and _med_col in df.columns:
                    df[_acs_col] = df[_acs_col].fillna(df[_med_col])
                elif _med_col in df.columns:
                    df.rename(columns={_med_col: _acs_col}, inplace=True)
            df.drop(
                columns=[
                    c
                    for c in (
                        "borough",
                        "median_income_med",
                        "population_med",
                        "rent_burden_med",
                    )
                    if c in df.columns
                ],
                inplace=True,
            )

        # 11. Static features — only if source columns present
        if "median_income_static" in df.columns and "median_income" in df.columns:
            df["median_income_static"] = df["median_income_static"].fillna(
                df["median_income"]
            )
        if "rent_pressure" in df.columns and "rent_burden" in df.columns:
            _max_rb = df["rent_burden"].max()
            if _max_rb > 0:
                df["rent_pressure"] = df["rent_pressure"].fillna(
                    df["rent_burden"] / _max_rb
                )
        if "mean_assessed_value" in df.columns and "median_income" in df.columns:
            df["mean_assessed_value"] = df["mean_assessed_value"].fillna(
                df["median_income"] * 12
            )

        # 12. Final fallback
        numeric_medians = df.select_dtypes(exclude="object").median()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(numeric_medians.get(col, df[col].median()))

    return df


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
            dedupe_columns = [text_col]  # pragma: no cover
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

    year_start = int(MODEL_CONFIG.get("temporal_data_start_year", 2020))
    year_end = int(MODEL_CONFIG.get("temporal_data_end_year", 2024))
    if "time_key" in frame.columns:
        time_key = pd.to_numeric(frame["time_key"], errors="coerce")
        frame = frame[time_key.between(year_start, year_end, inclusive="both")].copy()

    present_key_columns = [column for column in key_columns if column in frame.columns]
    if present_key_columns:
        frame = frame.drop_duplicates(subset=present_key_columns, keep="last")

    frame = frame[frame[target_col].notna()].copy()

    if "label_quality" in frame.columns:
        frame = frame[frame["label_quality"].fillna(0.0) >= min_label_quality].copy()

    # Zone-time NLP aggregates may include string labels;
    # learned scorer uses numeric cols only.
    _string_feature_drop = frozenset({"dominant_subtype"})
    _drop = [c for c in _string_feature_drop if c in frame.columns]
    if _drop:
        frame = frame.drop(columns=_drop)

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
            "Training frame contains non-numeric feature columns that "
            "must be encoded first: " + ", ".join(non_numeric)
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
