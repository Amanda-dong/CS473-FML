from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


from src.utils import HALAL_CUISINES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "output"

YELP_REVIEWS = RAW / "yelp_reviews_with_zones.csv"
GEMINI_LABELS = RAW / "gemini_labels_full.csv"
HYGIENE = RAW / "restaurant_hygiene.csv"
PHASE1 = OUT_DIR / "phase1_cluster_assignments.csv"

JOIN_CANDIDATES = ["review_id", "restaurant_id", "business_id"]
LABEL_CANDIDATES = [
    "halal_label",
    "label",
    "gemini_label",
    "category",
    "halal_relevance",
]


def _load_yearly_nta_signals() -> pd.DataFrame:
    reviews = pd.read_csv(YELP_REVIEWS)
    gemini = pd.read_csv(GEMINI_LABELS)

    join_key = next(
        (c for c in JOIN_CANDIDATES if c in reviews.columns and c in gemini.columns),
        None,
    )
    label_col = next((c for c in LABEL_CANDIDATES if c in gemini.columns), None)
    if join_key is None or label_col is None:
        raise ValueError("Could not resolve Yelp/Gemini join key or label column.")

    joined = reviews.merge(
        gemini[[join_key, label_col]].drop_duplicates(subset=[join_key]),
        on=join_key,
        how="left",
    )
    joined["year"] = pd.to_datetime(joined["review_date"], errors="coerce").dt.year
    joined = joined.dropna(subset=["nta", "year"]).copy()

    label = joined[label_col].fillna("").astype(str).str.lower()
    joined["is_halal"] = label.str.contains("halal", case=False, regex=False).astype(
        int
    )
    joined["is_explicit"] = label.eq("explicit_halal").astype(int)

    agg = joined.groupby(["nta", "year"], as_index=False).agg(
        total_reviews=("review_id", "count"),
        halal_count=("is_halal", "sum"),
        explicit_count=("is_explicit", "sum"),
    )
    agg["halal_related_share"] = agg["halal_count"] / agg["total_reviews"]
    agg["explicit_halal_share"] = agg["explicit_count"] / agg["total_reviews"]

    global_mean = agg["halal_count"].sum() / agg["total_reviews"].sum()
    agg["shrunk_share"] = (agg["halal_count"] + 10 * global_mean) / (
        agg["total_reviews"] + 10
    )
    return agg.rename(columns={"nta": "nta_id"})


def build_forecast():
    yearly = _load_yearly_nta_signals()

    y2022 = yearly[(yearly["year"] == 2022) & (yearly["total_reviews"] >= 3)].copy()
    y2023 = yearly[(yearly["year"] == 2023) & (yearly["total_reviews"] >= 3)].copy()

    phase1 = pd.read_csv(PHASE1)[
        ["nta_id", "halal_supply_rate", "gap_score", "halal_cuisine_diversity"]
    ].copy()

    model_df = (
        y2022[
            [
                "nta_id",
                "shrunk_share",
                "explicit_halal_share",
                "total_reviews",
            ]
        ]
        .rename(
            columns={
                "shrunk_share": "halal_related_share_2022",
                "explicit_halal_share": "explicit_halal_share_2022",
                "total_reviews": "total_reviews_2022",
            }
        )
        .merge(
            y2023[["nta_id", "halal_related_share"]].rename(
                columns={"halal_related_share": "halal_related_share_2023"}
            ),
            on="nta_id",
            how="inner",
        )
        .merge(phase1, on="nta_id", how="inner")
        .dropna()
        .copy()
    )

    feature_cols = [
        "halal_related_share_2022",
        "explicit_halal_share_2022",
        "total_reviews_2022",
        "halal_supply_rate",
        "gap_score",
        "halal_cuisine_diversity",
    ]
    X = model_df[feature_cols]
    y = model_df["halal_related_share_2023"].astype(float)

    print(f"Forecast sample size after join: {len(model_df)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    model.fit(X, y)
    model_df["halal_demand_forecast"] = model.predict(X)

    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})

    ablation_rows = []
    for col in feature_cols:
        cols = [c for c in feature_cols if c != col]
        ab_model = Ridge(alpha=1.0)
        ab_scores = cross_val_score(ab_model, model_df[cols], y, cv=cv, scoring="r2")
        ablation_rows.append(
            {
                "dropped_feature": col,
                "r2_mean": ab_scores.mean(),
                "r2_std": ab_scores.std(),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)

    baseline_pred = model_df["halal_related_share_2022"].to_numpy()
    baseline_r2 = r2_score(y, baseline_pred)

    top_actual = model_df.nlargest(5, "halal_related_share_2023")[
        [
            "nta_id",
            "halal_related_share_2022",
            "halal_related_share_2023",
            "halal_demand_forecast",
        ]
    ]
    bottom_actual = model_df.nsmallest(5, "halal_related_share_2023")[
        [
            "nta_id",
            "halal_related_share_2022",
            "halal_related_share_2023",
            "halal_demand_forecast",
        ]
    ]

    forecast_df = model_df[["nta_id", "halal_demand_forecast"]].copy()
    diagnostics = {
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "baseline_r2": baseline_r2,
        "coefficients": coef_df,
        "ablation": ablation_df,
        "top_actual": top_actual,
        "bottom_actual": bottom_actual,
        "feature_cols": feature_cols,
    }
    return forecast_df, diagnostics


def build_entry_forecast():
    yearly = _load_yearly_nta_signals()
    phase1 = pd.read_csv(PHASE1)[
        [
            "nta_id",
            "demand_score",
            "gap_score",
            "halal_cuisine_diversity",
            "halal_supply_rate",
        ]
    ].copy()

    hygiene = pd.read_csv(HYGIENE)
    hygiene["INSPECTION DATE"] = pd.to_datetime(
        hygiene["INSPECTION DATE"], errors="coerce"
    )
    hygiene["year"] = hygiene["INSPECTION DATE"].dt.year
    hygiene = hygiene[hygiene["year"].between(2010, 2025)].copy()

    # Use shared HALAL_CUISINES (lowercase) — apply .str.lower() for CAMIS title-case data
    halal_mask = (
        hygiene["CUISINE DESCRIPTION"].str.strip().str.lower().isin(HALAL_CUISINES)
    )
    hygiene = hygiene[halal_mask].dropna(subset=["CAMIS", "NTA", "INSPECTION DATE"])

    first_seen = (
        hygiene.groupby("CAMIS", as_index=False)["INSPECTION DATE"]
        .min()
        .rename(columns={"INSPECTION DATE": "first_seen_date"})
    )
    first_seen["first_year"] = first_seen["first_seen_date"].dt.year
    camis_nta = (
        hygiene.sort_values("INSPECTION DATE")
        .drop_duplicates(subset=["CAMIS"])[["CAMIS", "NTA"]]
        .rename(columns={"NTA": "nta_id"})
    )
    new_halal = camis_nta.merge(
        first_seen[["CAMIS", "first_year"]], on="CAMIS", how="inner"
    )
    new_counts = (
        new_halal.groupby(["nta_id", "first_year"], as_index=False)["CAMIS"]
        .nunique()
        .rename(columns={"CAMIS": "new_halal_count"})
    )

    n2023 = new_counts[new_counts["first_year"] == 2023][
        ["nta_id", "new_halal_count"]
    ].rename(columns={"new_halal_count": "new_halal_count_2023"})
    n2024 = new_counts[new_counts["first_year"] == 2024][
        ["nta_id", "new_halal_count"]
    ].rename(columns={"new_halal_count": "new_halal_count_2024"})
    y2023 = yearly[(yearly["year"] == 2023) & (yearly["total_reviews"] >= 3)].copy()
    feature_year = y2023[
        ["nta_id", "shrunk_share", "explicit_halal_share", "total_reviews"]
    ].rename(
        columns={
            "shrunk_share": "halal_related_share_2023",
            "explicit_halal_share": "explicit_halal_share_2023",
            "total_reviews": "total_reviews_2023",
        }
    )

    model_df = (
        feature_year.merge(n2023, on="nta_id", how="inner")
        .merge(n2024, on="nta_id", how="inner")
        .merge(phase1, on="nta_id", how="inner")
        .dropna()
        .copy()
    )

    feature_cols = [
        "halal_related_share_2023",
        "explicit_halal_share_2023",
        "total_reviews_2023",
        "new_halal_count_2023",
        "demand_score",
        "gap_score",
        "halal_cuisine_diversity",
        "halal_supply_rate",
    ]
    X = model_df[feature_cols]
    y = model_df["new_halal_count_2024"].astype(float)

    print(f"Entry forecast sample size after join: {len(model_df)}")

    cv = KFold(n_splits=min(5, len(model_df)), shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    model.fit(X, y)
    model_df["new_halal_entry_forecast"] = pd.Series(
        model.predict(X), index=model_df.index
    ).clip(lower=0.0)

    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})

    ablation_rows = []
    for col in feature_cols:
        cols = [c for c in feature_cols if c != col]
        ab_model = Ridge(alpha=1.0)
        ab_scores = cross_val_score(ab_model, model_df[cols], y, cv=cv, scoring="r2")
        ablation_rows.append(
            {
                "dropped_feature": col,
                "r2_mean": ab_scores.mean(),
                "r2_std": ab_scores.std(),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)

    baseline_pred = model_df["new_halal_count_2023"].to_numpy()
    baseline_r2 = r2_score(y, baseline_pred)

    top_actual = model_df.nlargest(5, "new_halal_count_2024")[
        [
            "nta_id",
            "new_halal_count_2023",
            "new_halal_count_2024",
            "new_halal_entry_forecast",
        ]
    ]
    bottom_actual = model_df.nsmallest(5, "new_halal_count_2024")[
        [
            "nta_id",
            "new_halal_count_2023",
            "new_halal_count_2024",
            "new_halal_entry_forecast",
        ]
    ]

    forecast_df = model_df[["nta_id", "new_halal_entry_forecast"]].copy()
    diagnostics = {
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "baseline_r2": baseline_r2,
        "coefficients": coef_df,
        "ablation": ablation_df,
        "top_actual": top_actual,
        "bottom_actual": bottom_actual,
        "sample_size": len(model_df),
    }
    return forecast_df, diagnostics
