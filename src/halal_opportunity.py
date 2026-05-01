from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import HALAL_CUISINES, minmax as _minmax


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HYGIENE = RAW / "restaurant_hygiene.csv"


def build_supply() -> pd.DataFrame:
    df = pd.read_csv(HYGIENE)
    df = df.rename(columns={"NTA": "nta_id"})
    df = df.dropna(subset=["nta_id", "CAMIS"]).copy()
    df = df.drop_duplicates(subset=["CAMIS"]).copy()
    cuisine = df["CUISINE DESCRIPTION"].fillna("").astype(str).str.strip().str.lower()
    df["is_halal"] = cuisine.isin(HALAL_CUISINES).astype(int)

    grouped = df.groupby("nta_id", as_index=False).agg(
        total_restaurants=("CAMIS", "count"),
        halal_restaurants=("is_halal", "sum"),
    )
    grouped["halal_supply_rate"] = (
        grouped["halal_restaurants"] / grouped["total_restaurants"]
    )
    halal_diversity = (
        df[df["is_halal"] == 1]
        .groupby("nta_id", as_index=False)["CUISINE DESCRIPTION"]
        .nunique()
        .rename(columns={"CUISINE DESCRIPTION": "halal_cuisine_diversity"})
    )
    grouped = grouped.merge(halal_diversity, on="nta_id", how="left")
    grouped["halal_cuisine_diversity"] = (
        grouped["halal_cuisine_diversity"].fillna(0).astype(float)
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


def build_gap(demand_df: pd.DataFrame, supply_df: pd.DataFrame) -> pd.DataFrame:
    merged = demand_df.merge(supply_df, on="nta_id", how="inner")
    merged["supply_norm"] = _minmax(merged["halal_supply_rate"])
    merged["gap_score"] = (merged["demand_score"] - merged["supply_norm"]).clip(lower=0)
    merged["gap_score"] = _minmax(merged["gap_score"])
    print(f"NTAs after demand/supply join: {len(merged)}")
    print(f"Mean gap_score: {merged['gap_score'].mean():.4f}")
    return merged
