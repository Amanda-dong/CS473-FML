from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.config import CFG
from src.utils import HALAL_CUISINES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "output"
INSPECTIONS = ROOT / "data" / "processed" / "inspections.parquet"

YELP_REVIEWS = RAW / "yelp_reviews_with_zones.csv"
GEMINI_LABELS = RAW / "gemini_labels_full.csv"
PHASE1 = OUT_DIR / "phase1_cluster_assignments.csv"

JOIN_CANDIDATES = ["review_id", "restaurant_id", "business_id"]
LABEL_CANDIDATES = [
    "halal_label",
    "label",
    "gemini_label",
    "category",
    "halal_relevance",
]


@functools.lru_cache(maxsize=1)
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
    agg["shrunk_share"] = (agg["halal_count"] + CFG.demand_prior * global_mean) / (
        agg["total_reviews"] + CFG.demand_prior
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

    cv = KFold(n_splits=CFG.ridge_cv_folds, shuffle=True, random_state=CFG.ridge_random_state)
    model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=cv)
    model.fit(X, y)
    best_alpha = float(model.alpha_)
    model_df["halal_demand_forecast"] = model.predict(X)

    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})

    ablation_rows = []
    for col in feature_cols:
        cols = [c for c in feature_cols if c != col]
        ab_model = Ridge(alpha=CFG.ridge_alpha)
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
        "r2_insample": r2_score(y, model.predict(X)),
        "r2_std": 0.0,
        "best_alpha": best_alpha,
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

    df_insp = pd.read_parquet(INSPECTIONS)
    df_insp["inspection_date"] = pd.to_datetime(df_insp["inspection_date"], errors="coerce")
    df_insp["year"] = df_insp["inspection_date"].dt.year
    df_insp = df_insp[df_insp["year"].between(2010, 2025)].copy()
    df_insp["cuisine_lower"] = df_insp["cuisine_type"].fillna("").str.strip().str.lower()
    halal_insp = df_insp[df_insp["cuisine_lower"].isin(CFG.halal_cuisines)].dropna(
        subset=["restaurant_id", "nta_id", "inspection_date"]
    )
    first_seen = (
        halal_insp.groupby("restaurant_id", as_index=False)["inspection_date"]
        .min()
        .rename(columns={"inspection_date": "first_seen_date"})
    )
    first_seen["first_year"] = first_seen["first_seen_date"].dt.year
    camis_nta = halal_insp.sort_values("inspection_date").drop_duplicates("restaurant_id")[
        ["restaurant_id", "nta_id"]
    ]
    new_halal = camis_nta.merge(
        first_seen[["restaurant_id", "first_year"]], on="restaurant_id", how="inner"
    )
    new_counts = (
        new_halal.groupby(["nta_id", "first_year"], as_index=False)["restaurant_id"]
        .nunique()
        .rename(columns={"restaurant_id": "new_halal_count"})
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

    if len(model_df) < CFG.ridge_cv_folds:
        print(f'Insufficient samples ({len(model_df)}) for entry forecast — using halal presence fallback')
        phase1_full = pd.read_csv(OUT_DIR / 'phase1_cluster_assignments.csv') if (OUT_DIR / 'phase1_cluster_assignments.csv').exists() else pd.DataFrame(columns=['nta_id','halal_supply_rate'])
        fallback = phase1_full[['nta_id']].copy() if 'nta_id' in phase1_full.columns else pd.DataFrame(columns=['nta_id'])
        fallback['new_halal_entry_forecast'] = 0.0
        coef_df = pd.DataFrame({'feature': feature_cols, 'coefficient': [0.0]*len(feature_cols)})
        ablation_rows = [{'feature': c, 'r2_mean': 0.0, 'r2_std': 0.0} for c in feature_cols]
        ablation_df = pd.DataFrame(ablation_rows)
        diag = {'r2_mean': 0.0, 'r2_std': 0.0, 'baseline_r2': 0.0, 'coefficients': coef_df, 'ablation': ablation_df, 'top_actual': pd.DataFrame(), 'bottom_actual': pd.DataFrame()}
        return fallback, diag

    cv = KFold(n_splits=CFG.ridge_cv_folds, shuffle=True, random_state=CFG.ridge_random_state)
    model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=cv)
    model.fit(X, y)
    best_alpha = float(model.alpha_)
    model_df["new_halal_entry_forecast"] = pd.Series(
        model.predict(X), index=model_df.index
    ).clip(lower=0.0)

    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})

    ablation_rows = []
    for col in feature_cols:
        cols = [c for c in feature_cols if c != col]
        ab_model = Ridge(alpha=CFG.ridge_alpha)
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
        "r2_insample": r2_score(y, model.predict(X)),
        "r2_std": 0.0,
        "best_alpha": best_alpha,
        "baseline_r2": baseline_r2,
        "coefficients": coef_df,
        "ablation": ablation_df,
        "top_actual": top_actual,
        "bottom_actual": bottom_actual,
        "sample_size": len(model_df),
    }
    return forecast_df, diagnostics
