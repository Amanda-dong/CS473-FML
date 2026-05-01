from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CFG
from src.utils import minmax as _minmax


ROOT = Path(__file__).resolve().parents[1]
INSPECTIONS = ROOT / "data" / "processed" / "inspections.parquet"


def build_supply(cfg=CFG):
    df = pd.read_parquet(INSPECTIONS)
    # Ensure columns exist and handle types
    df["cuisine_lower"] = (
        df["cuisine_type"].fillna("").astype(str).str.strip().str.lower()
    )
    df["is_halal"] = df["cuisine_lower"].isin(cfg.halal_cuisines).astype(int)

    # Deduplicate by restaurant_id keeping most recent inspection
    df = (
        df.sort_values("inspection_date", ascending=False)
        .drop_duplicates("restaurant_id")
        .copy()
    )
    df = df.dropna(subset=["nta_id"]).copy()

    # Compute metrics
    grouped = df.groupby("nta_id", as_index=False).agg(
        total_restaurants=("restaurant_id", "count"),
        halal_restaurants=("is_halal", "sum"),
    )
    grouped["halal_supply_rate"] = (
        grouped["halal_restaurants"] / grouped["total_restaurants"]
    )

    # Compute halal_cuisine_diversity
    halal_diversity = (
        df[df["is_halal"] == 1]
        .groupby("nta_id", as_index=False)["cuisine_type"]
        .nunique()
        .rename(columns={"cuisine_type": "halal_cuisine_diversity"})
    )
    grouped = grouped.merge(halal_diversity, on="nta_id", how="left")
    grouped["halal_cuisine_diversity"] = (
        grouped["halal_cuisine_diversity"].fillna(0.0).astype(float)
    )

    return grouped[
        [
            "nta_id",
            "total_restaurants",
            "halal_restaurants",
            "halal_supply_rate",
            "halal_cuisine_diversity",
        ]
    ].copy()


def build_gap(demand_df, supply_df, cfg=CFG):
    merged = demand_df.merge(supply_df, on="nta_id", how="inner")
    merged["supply_norm"] = _minmax(merged["halal_supply_rate"])

    # Combined demand calc
    merged["combined_demand"] = (
        cfg.gap_demand_blend * merged["demand_score"]
        + cfg.gap_latent_blend * merged["latent_demand_score"]
    )

    # Use combined_demand instead of demand_score for gap_score calc
    merged["gap_score"] = (merged["combined_demand"] - merged["supply_norm"]).clip(
        lower=0
    )
    merged["gap_score"] = _minmax(merged["gap_score"])

    # add halal_cuisine_diversity_norm
    merged["halal_cuisine_diversity_norm"] = _minmax(merged["halal_cuisine_diversity"])

    print(f"NTAs after demand/supply join: {len(merged)}")
    print(f"Mean gap_score: {merged['gap_score'].mean():.4f}")

    return merged
