"""Enrich zone_features.parquet with NTA cuisine diversity and Yelp rating signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def _cuisine_diversity_features(inspections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cuisine_type from inspections to NTA level.

    Returns DataFrame with columns:
        zone_id, cuisine_diversity, dominant_cuisine, high_risk_cuisine_share
    """
    df = inspections.dropna(subset=["nta_id", "cuisine_type"]).copy()
    df = df[df["nta_id"].str.len() == 6]  # keep standard 2020 6-char NTA codes only

    # High-risk cuisines (historically higher violation rates)
    high_risk = {
        "chinese",
        "mexican",
        "american",
        "latin american",
        "caribbean",
        "bakery products/desserts",
        "spanish",
    }

    records = []
    for nta_id, grp in df.groupby("nta_id"):
        counts = grp["cuisine_type"].value_counts()
        total = counts.sum()
        probs = counts / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        # Normalize to [0, 1] by max entropy (log of n unique cuisines)
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        cuisine_diversity = float(entropy / max_entropy)

        dominant = str(counts.idxmax()).lower()
        high_risk_share = float(
            grp["cuisine_type"].str.lower().isin(high_risk).sum() / total
        )

        records.append(
            {
                "zone_id": nta_id,
                "cuisine_diversity": round(cuisine_diversity, 4),
                "dominant_cuisine": dominant,
                "high_risk_cuisine_share": round(high_risk_share, 4),
            }
        )

    return pd.DataFrame(records)


def _yelp_nta_features(yelp_zones: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Yelp ratings from already-zoned review data to NTA level.

    yelp_reviews_with_zones.csv has an `nta` column in standard 6-char 2020 format.

    Returns DataFrame with columns:
        zone_id, yelp_avg_rating, yelp_review_density
    """
    df = yelp_zones.dropna(subset=["nta", "rating"]).copy()
    df = df[df["nta"].str.len() == 6]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    agg = (
        df.groupby("nta")
        .agg(
            yelp_avg_rating=("rating", "mean"),
            yelp_review_count=("rating", "count"),
        )
        .reset_index()
        .rename(columns={"nta": "zone_id"})
    )
    # Normalize review count to a 0-1 density score
    max_count = agg["yelp_review_count"].max()
    agg["yelp_review_density"] = (agg["yelp_review_count"] / max(max_count, 1)).round(4)
    agg["yelp_avg_rating"] = agg["yelp_avg_rating"].round(4)
    return agg[["zone_id", "yelp_avg_rating", "yelp_review_density"]]


def main() -> None:
    zone_features_path = PROCESSED_DIR / "zone_features.parquet"
    inspections_path = PROCESSED_DIR / "inspections.parquet"
    yelp_zones_path = Path("data/raw") / "yelp_reviews_with_zones.csv"

    zf = pd.read_parquet(zone_features_path)
    print(f"zone_features: {zf.shape} zones, cols: {list(zf.columns)}")

    inspections = pd.read_parquet(inspections_path)
    cuisine_feats = _cuisine_diversity_features(inspections)
    print(f"Cuisine diversity: {len(cuisine_feats)} NTAs")

    yelp_zones = pd.read_csv(yelp_zones_path)
    yelp_feats = _yelp_nta_features(yelp_zones)
    print(f"Yelp NTA features: {len(yelp_feats)} NTAs")

    # Drop any old versions of these columns before merging
    drop_cols = [
        c
        for c in zf.columns
        if c
        in {
            "cuisine_diversity",
            "dominant_cuisine",
            "high_risk_cuisine_share",
            "yelp_avg_rating",
            "yelp_review_density",
        }
    ]
    if drop_cols:
        zf = zf.drop(columns=drop_cols)

    zf = zf.merge(cuisine_feats, on="zone_id", how="left")
    zf = zf.merge(yelp_feats, on="zone_id", how="left")

    # Fill NAs: cuisine diversity → 0 (unknown), yelp → median
    zf["cuisine_diversity"] = zf["cuisine_diversity"].fillna(0.0)
    zf["high_risk_cuisine_share"] = zf["high_risk_cuisine_share"].fillna(0.0)
    zf["dominant_cuisine"] = zf["dominant_cuisine"].fillna("unknown")
    zf["yelp_avg_rating"] = zf["yelp_avg_rating"].fillna(zf["yelp_avg_rating"].median())
    zf["yelp_review_density"] = zf["yelp_review_density"].fillna(0.0)

    zf.to_parquet(zone_features_path, index=False)
    print(f"Saved enriched zone_features: {zf.shape}")
    print(
        "New cols: cuisine_diversity, dominant_cuisine, high_risk_cuisine_share, "
        "yelp_avg_rating, yelp_review_density"
    )
    print(f"NTAs with cuisine data: {zf['cuisine_diversity'].gt(0).sum()}")
    print(f"NTAs with Yelp data: {zf['yelp_review_density'].gt(0).sum()}")


if __name__ == "__main__":
    main()
