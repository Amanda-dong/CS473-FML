from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import minmax as _minmax


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

YELP_REVIEWS = RAW / "yelp_reviews_with_zones.csv"
GEMINI_LABELS = RAW / "gemini_labels_full.csv"

LABEL_CANDIDATES = [
    "halal_label",
    "label",
    "gemini_label",
    "category",
    "label_type",
    "halal_relevance",
]
JOIN_CANDIDATES = ["review_id", "restaurant_id", "business_id"]


def build_demand() -> pd.DataFrame:
    reviews = pd.read_csv(YELP_REVIEWS)
    labels = pd.read_csv(GEMINI_LABELS)

    print("Gemini columns:", labels.columns.tolist())

    join_key = next(
        (c for c in JOIN_CANDIDATES if c in reviews.columns and c in labels.columns),
        None,
    )
    if join_key is None:
        raise ValueError(
            "No shared join key found between Yelp reviews and Gemini labels."
        )

    label_col = next((c for c in LABEL_CANDIDATES if c in labels.columns), None)

    merged = reviews.copy()
    merged["review_date"] = pd.to_datetime(merged["review_date"], errors="coerce")
    reference_year = 2024
    merged["review_year"] = merged["review_date"].dt.year.fillna(reference_year)
    merged["decay_weight"] = (0.85 ** (reference_year - merged["review_year"])).clip(
        lower=0.1
    )
    merged.loc[merged["review_date"].isna(), "decay_weight"] = 1.0

    if label_col is not None:
        merged = merged.merge(
            labels[[join_key, label_col]].drop_duplicates(subset=[join_key]),
            on=join_key,
            how="left",
        )
        normalized = merged[label_col].fillna("").astype(str).str.lower()
        merged["is_halal"] = normalized.str.contains(
            "halal", case=False, regex=False
        ).astype(int)
        merged["is_explicit"] = normalized.eq("explicit_halal").astype(int)
    else:
        text = merged["review_text"].fillna("").astype(str).str.lower()
        merged["is_halal"] = text.str.contains("halal", case=False, regex=False).astype(
            int
        )
        merged["is_explicit"] = 0

    merged["is_halal_weighted"] = merged["is_halal"] * merged["decay_weight"]
    merged["is_explicit_weighted"] = merged["is_explicit"] * merged["decay_weight"]

    merged = merged.dropna(subset=["nta"]).copy()

    grouped = merged.groupby("nta", as_index=False).agg(
        total_reviews=("review_id", "count"),
        halal_count=("is_halal_weighted", "sum"),
        explicit_count=("is_explicit_weighted", "sum"),
    )
    global_mean = grouped["halal_count"].sum() / grouped["total_reviews"].sum()
    prior = 10.0
    grouped["halal_related_share"] = grouped["halal_count"] / grouped["total_reviews"]
    grouped["explicit_halal_share"] = (
        grouped["explicit_count"] / grouped["total_reviews"]
    )
    grouped["shrunk_share"] = (grouped["halal_count"] + prior * global_mean) / (
        grouped["total_reviews"] + prior
    )
    grouped["demand_score"] = _minmax(grouped["shrunk_share"])
    grouped["review_count_flag"] = grouped["total_reviews"].apply(
        lambda x: "low confidence" if x < 30 else "high confidence"
    )
    grouped = grouped.rename(columns={"nta": "nta_id"})

    top3 = grouped.nlargest(3, "demand_score")[["nta_id", "demand_score"]]
    print(f"Demand NTAs: {len(grouped)}")
    print(f"Mean demand_score: {grouped['demand_score'].mean():.4f}")
    print("Top 3 NTAs by demand_score:")
    print(top3.to_string(index=False))

    return grouped[
        [
            "nta_id",
            "total_reviews",
            "halal_related_share",
            "explicit_halal_share",
            "shrunk_share",
            "demand_score",
            "review_count_flag",
        ]
    ].copy()
