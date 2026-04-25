"""Composite ground-truth label for zone attractiveness."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.zone_crosswalk import aggregate_nta_to_zone


def _license_entity_ids(licenses_df: pd.DataFrame) -> pd.Series:
    """Return the best available per-business identifier from license data."""
    for column in ("restaurant_id", "business_unique_id"):
        if column not in licenses_df.columns:
            continue
        ids = licenses_df[column].replace({"UNKNOWN": pd.NA, "": pd.NA})
        if ids.notna().any():
            return ids.astype("string")
    return pd.Series(pd.NA, index=licenses_df.index, dtype="string")


def _survival_rate(licenses_df: pd.DataFrame, horizon_years: int = 2) -> pd.DataFrame:
    """Fraction of restaurants opened in year T still active at T+horizon.

    For each cohort opened in year T, check whether each restaurant appears
    with an Active/Issued status in data from year T+horizon.  If the dataset
    does not extend to T+horizon, the cohort is excluded (not filled with 0.5).
    """
    if licenses_df.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_survival"])

    df = licenses_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["_entity_id"] = _license_entity_ids(df)
    df = df.dropna(subset=["event_date", "_entity_id"])
    df["year"] = df["event_date"].dt.year

    max_data_year = int(df["year"].max())

    # Identify first appearance year per restaurant
    first_seen = (
        df.groupby("_entity_id")
        .agg(open_year=("year", "min"), zone_id=("nta_id", "first"))
        .reset_index()
        .rename(columns={"_entity_id": "restaurant_id"})
    )

    # Only keep cohorts where T+horizon is observable in the data
    first_seen = first_seen[first_seen["open_year"] + horizon_years <= max_data_year]

    if first_seen.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_survival"])

    # Check if restaurant has an Active/Issued record in year T+horizon
    active_statuses = frozenset({"Active", "Issued"})
    first_seen["target_year"] = first_seen["open_year"] + horizon_years

    # Build a set of (restaurant_id, year) pairs with active status
    active_records = df[df["license_status"].isin(active_statuses)]
    active_pairs = set(zip(active_records["_entity_id"], active_records["year"]))

    first_seen["survived"] = [
        int((rid, ty) in active_pairs)
        for rid, ty in zip(first_seen["restaurant_id"], first_seen["target_year"])
    ]

    # Group by zone and open_year
    result = (
        first_seen.groupby(["zone_id", "open_year"])
        .agg(y_survival=("survived", "mean"))
        .reset_index()
        .rename(columns={"open_year": "time_key"})
    )

    return result[["zone_id", "time_key", "y_survival"]]


def _review_quality(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Average review rating per zone-year, GLOBAL z-score then sigmoid to [0,1]."""
    if reviews_df.empty or "rating" not in reviews_df.columns:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_review_quality"])
    if "zone_id" not in reviews_df.columns or "time_key" not in reviews_df.columns:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_review_quality"])

    grouped = (
        reviews_df.groupby(["zone_id", "time_key"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "mean_rating"})
    )

    # Global z-score across all years
    mu = grouped["mean_rating"].mean()
    sigma = grouped["mean_rating"].std()
    if sigma == 0 or pd.isna(sigma):
        sigma = 1.0
    z = (grouped["mean_rating"] - mu) / sigma

    # Sigmoid
    grouped["y_review_quality"] = 1.0 / (1.0 + np.exp(-z))

    return grouped[["zone_id", "time_key", "y_review_quality"]]


def _license_velocity_signal(licenses_df: pd.DataFrame) -> pd.DataFrame:
    """Net new healthy restaurant licenses, percentile-ranked across all years."""
    if licenses_df.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_license_velocity"])

    df = licenses_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "nta_id"])
    df["year"] = df["event_date"].dt.year.astype(int)

    open_statuses = frozenset({"Active", "Issued"})
    close_statuses = frozenset({"Inactive", "Revoked", "Expired"})

    df["_open"] = df["license_status"].isin(open_statuses).astype(int)
    df["_close"] = df["license_status"].isin(close_statuses).astype(int)

    grouped = df.groupby(["nta_id", "year"], as_index=False).agg(
        net=("_open", "sum"),
        closes=("_close", "sum"),
    )
    grouped["net_velocity"] = grouped["net"] - grouped["closes"]

    # Global percentile rank across all zone-years
    grouped["y_license_velocity"] = grouped["net_velocity"].rank(pct=True)

    result = grouped.rename(columns={"nta_id": "zone_id", "year": "time_key"})
    return result[["zone_id", "time_key", "y_license_velocity"]]


def _inspection_quality(inspections_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of restaurants with grade A per zone-year."""
    if inspections_df.empty:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_inspection"])

    df = inspections_df.copy()

    # Inspections ETL outputs nta_id, not zone_id
    id_col = "nta_id" if "nta_id" in df.columns else ("zone_id" if "zone_id" in df.columns else None)
    if "grade" not in df.columns or id_col is None:
        return pd.DataFrame(columns=["zone_id", "time_key", "y_inspection"])

    if "year" not in df.columns and "time_key" in df.columns:
        df["year"] = df["time_key"]
    elif "year" not in df.columns and "inspection_date" in df.columns:
        df["year"] = pd.to_datetime(df["inspection_date"], errors="coerce").dt.year

    df = df.dropna(subset=["year", id_col])
    df["year"] = df["year"].astype(int)
    df["is_a"] = (df["grade"] == "A").astype(int)

    result = (
        df.groupby([id_col, "year"])
        .agg(y_inspection=("is_a", "mean"))
        .reset_index()
        .rename(columns={id_col: "zone_id", "year": "time_key"})
    )

    return result[["zone_id", "time_key", "y_inspection"]]


def build_ground_truth(
    licenses_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    inspections_df: pd.DataFrame,
    crosswalk: dict[str, list[str]] | None = None,
    weights: tuple[float, ...] = (0.35, 0.25, 0.20, 0.20),
) -> pd.DataFrame:
    """Build composite zone attractiveness label.

    Features measured at time T, outcome at T+2.
    Returns: zone_id, time_key, y_survival, y_review_quality,
             y_license_velocity, y_inspection, y_composite,
             missingness_fraction, label_quality
    """
    output_cols = [
        "zone_id", "time_key", "y_survival", "y_review_quality",
        "y_license_velocity", "y_inspection", "y_composite",
        "missingness_fraction", "label_quality",
    ]

    # Components are computed at NTA level (zone_id = NTA code).
    # Aggregate each to micro-zone level via crosswalk before merging.
    def _to_zone(df: pd.DataFrame, value_col: str, agg: str = "mean") -> pd.DataFrame:
        if df.empty:
            return df
        renamed = df.rename(columns={"zone_id": "nta_id"})
        return aggregate_nta_to_zone(renamed, zone_col="nta_id", agg_rules={value_col: agg})

    surv = _to_zone(_survival_rate(licenses_df), "y_survival")
    review_q = _to_zone(_review_quality(reviews_df), "y_review_quality")
    lic_vel = _to_zone(_license_velocity_signal(licenses_df), "y_license_velocity")
    insp_q = _to_zone(_inspection_quality(inspections_df), "y_inspection")

    # Start from survival as base, merge others
    if surv.empty:
        return pd.DataFrame(columns=output_cols)

    result = surv.copy()
    for component in [review_q, lic_vel, insp_q]:
        if not component.empty:
            result = result.merge(component, on=["zone_id", "time_key"], how="left")

    component_cols = ["y_survival", "y_review_quality", "y_license_velocity", "y_inspection"]
    w = np.array(weights, dtype=float)

    # Ensure columns exist (as NaN, not filled)
    for col in component_cols:
        if col not in result.columns:
            result[col] = np.nan

    # Track missingness
    component_matrix = result[component_cols]
    present_mask = component_matrix.notna()
    n_components = len(component_cols)

    result["missingness_fraction"] = 1.0 - present_mask.sum(axis=1) / n_components
    result["label_quality"] = present_mask.sum(axis=1) / n_components

    # Weighted average using only available components (re-weight to sum to 1.0)
    values = component_matrix.values  # (n, 4)
    weights_matrix = np.where(present_mask.values, w[np.newaxis, :], 0.0)
    weight_sums = weights_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows with no components
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    normalized_weights = weights_matrix / weight_sums

    # Replace NaN with 0 for the multiplication (masked out by weight=0 anyway)
    safe_values = np.where(present_mask.values, values, 0.0)
    result["y_composite"] = (normalized_weights * safe_values).sum(axis=1)

    # Rows with zero present components get NaN composite
    all_missing = present_mask.sum(axis=1) == 0
    result.loc[all_missing, "y_composite"] = np.nan

    return result[output_cols]
