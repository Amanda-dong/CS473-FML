from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from numpy import log1p
from scipy.stats import beta as _beta_dist

from src.config import CFG, ModelConfig
from src.utils import minmax as _minmax


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

YELP_REVIEWS = RAW / "yelp_reviews_with_zones.csv"
GEMINI_LABELS = RAW / "gemini_labels_full.csv"
NTA_DEMOGRAPHICS = RAW / "nta_demographics_processed.csv"

LABEL_CANDIDATES = [
    "halal_label",
    "label",
    "gemini_label",
    "category",
    "label_type",
    "halal_relevance",
]
JOIN_CANDIDATES = ["review_id", "restaurant_id", "business_id"]


def _load_raw_data():
    reviews = pd.read_csv(YELP_REVIEWS)
    labels = pd.read_csv(GEMINI_LABELS)

    join_key = next(
        (c for c in JOIN_CANDIDATES if c in reviews.columns and c in labels.columns),
        None,
    )
    if join_key is None:
        raise ValueError(
            "No shared join key found between Yelp reviews and Gemini labels."
        )

    label_col = next((c for c in LABEL_CANDIDATES if c in labels.columns), None)
    return reviews, labels, join_key, label_col


def build_latent_demand(reviews, labels, join_key, label_col, cfg=CFG):
    merged = reviews.merge(
        labels[[join_key, label_col]].drop_duplicates(subset=[join_key]),
        on=join_key,
        how="left",
    )
    normalized = merged[label_col].fillna("").astype(str).str.lower()
    is_halal = normalized.str.contains("halal", case=False, regex=False).astype(int)
    is_explicit = normalized.eq("explicit_halal").astype(int)
    is_implicit = (is_halal - is_explicit).clip(lower=0)

    keyword_pattern = re.compile(
        "|".join(re.escape(k) for k in cfg.halal_keywords), re.IGNORECASE
    )
    has_keyword = (
        reviews["review_text"]
        .fillna("")
        .apply(lambda t: bool(keyword_pattern.search(str(t))))
        .astype(int)
    )

    df = merged.copy()
    df["is_implicit"] = is_implicit
    df["has_keyword"] = has_keyword
    df = df.dropna(subset=["nta"])

    grouped = df.groupby("nta", as_index=False).agg(
        total_reviews=("review_id", "count"),
        implicit_count=("is_implicit", "sum"),
        kw_count=("has_keyword", "sum"),
    )

    grouped["implicit_rate"] = grouped["implicit_count"] / grouped["total_reviews"]
    grouped["keyword_density"] = grouped["kw_count"] / grouped["total_reviews"]

    max_rev = grouped["total_reviews"].max()
    max_rev = max_rev if max_rev > 0 else 1
    grouped["activity_score"] = log1p(grouped["total_reviews"]) / log1p(max_rev)

    raw_latent = (
        cfg.latent_implicit_weight * _minmax(grouped["implicit_rate"])
        + cfg.latent_keyword_weight * _minmax(grouped["keyword_density"])
        + cfg.latent_activity_weight * grouped["activity_score"]
    )
    grouped["latent_demand_score"] = _minmax(raw_latent)
    grouped = grouped.rename(columns={"nta": "nta_id"})

    return grouped[
        [
            "nta_id",
            "implicit_count",
            "implicit_rate",
            "kw_count",
            "keyword_density",
            "activity_score",
            "latent_demand_score",
        ]
    ]


def build_demand() -> pd.DataFrame:
    reviews, labels, join_key, label_col = _load_raw_data()

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
    prior = CFG.demand_prior
    grouped["halal_related_share"] = grouped["halal_count"] / grouped["total_reviews"]
    grouped["explicit_halal_share"] = (
        grouped["explicit_count"] / grouped["total_reviews"]
    )
    grouped["shrunk_share"] = (grouped["halal_count"] + prior * global_mean) / (
        grouped["total_reviews"] + prior
    )
    grouped["demand_score"] = _minmax(grouped["shrunk_share"])
    grouped["review_count_flag"] = grouped["total_reviews"].apply(
        lambda x: "low confidence" if x < CFG.low_confidence_threshold else "high confidence"
    )
    grouped = grouped.rename(columns={"nta": "nta_id"})

    # Join population for per-capita demand and Bayesian confidence intervals
    if NTA_DEMOGRAPHICS.exists():
        pop_df = pd.read_csv(NTA_DEMOGRAPHICS)
        grouped = grouped.merge(pop_df, on="nta_id", how="left")
        grouped["population"] = grouped["population"].fillna(grouped["population"].median())
    else:
        grouped["population"] = 0.0
    grouped["demand_per_capita"] = (
        (grouped["halal_count"] / grouped["population"].replace(0, pd.NA)) * 1000
    ).fillna(0.0).clip(lower=0.0)
    # Bayesian Beta credible intervals (80%) for halal share
    h = grouped['halal_count'].clip(lower=0)
    n = grouped["total_reviews"].clip(lower=1)
    grouped["demand_ci_lo"] = _beta_dist.ppf(0.1, h + 1, n - h + 1)
    grouped["demand_ci_hi"] = _beta_dist.ppf(0.9, h + 1, n - h + 1)

    latent_df = build_latent_demand(reviews, labels, join_key, label_col)
    grouped = grouped.merge(
        latent_df[["nta_id", "latent_demand_score"]], on="nta_id", how="left"
    )
    grouped["latent_demand_score"] = grouped["latent_demand_score"].fillna(0.0)

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
            "latent_demand_score",
            "demand_per_capita",
            "population",
            "demand_ci_lo",
            "demand_ci_hi",
            "review_count_flag",
        ]
    ].copy()
