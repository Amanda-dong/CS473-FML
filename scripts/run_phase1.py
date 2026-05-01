from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.halal_demand import build_demand
from src.halal_opportunity import build_gap, build_supply
from src.halal_kmeans import HalalKMeans, run_kmeans


OUT_DIR = ROOT / "data" / "output"


def main() -> None:
    demand = build_demand()
    supply = build_supply()
    merged = build_gap(demand, supply)

    feature_cols = ['demand_score', 'latent_demand_score', 'halal_supply_rate', 'halal_cuisine_diversity_norm']

    elbow_df = merged.dropna(subset=feature_cols).copy()
    means_all = elbow_df[feature_cols].mean()
    stds_all = elbow_df[feature_cols].std().replace(0, 1.0)
    X_all = ((elbow_df[feature_cols] - means_all) / stds_all).to_numpy(dtype=float)
    elbow_rows = []
    prev_inertia = None
    for k in range(2, 9):
        km_tmp = HalalKMeans(k=k, random_state=42)
        km_tmp.fit(X_all)
        sil_tmp = silhouette_score(X_all, km_tmp.labels_)
        drop = None if prev_inertia is None else prev_inertia - km_tmp.inertia_
        pct_drop = None if prev_inertia is None else (drop / prev_inertia) * 100
        elbow_rows.append(
            {
                "k": k,
                "inertia": km_tmp.inertia_,
                "drop_from_prev": drop,
                "pct_drop_from_prev": pct_drop,
                "silhouette": sil_tmp,
            }
        )
        prev_inertia = km_tmp.inertia_
    elbow_table = pd.DataFrame(elbow_rows)
    print("Elbow / silhouette table:")
    print(
        elbow_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN",
        )
    )

    clustered, km = run_kmeans(merged, feature_cols, k=4)

    means = clustered[feature_cols].mean()
    stds = clustered[feature_cols].std().replace(0, 1.0)
    Xz = ((clustered[feature_cols] - means) / stds).to_numpy(dtype=float)
    sil = silhouette_score(Xz, clustered["cluster_id"].to_numpy())

    size_df = (
        clustered["market_type"]
        .value_counts()
        .rename_axis("market_type")
        .reset_index(name="nta_count")
    )
    centroid_df = (
        clustered.groupby("market_type", as_index=False)[feature_cols]
        .mean()
        .rename(
            columns={
                "demand_score": "demand_score_mean",
                "latent_demand_score": "latent_demand_score_mean",
                "halal_supply_rate": "halal_supply_rate_mean",
                "halal_cuisine_diversity_norm": "halal_cuisine_diversity_norm_mean",
            }
        )
        .merge(size_df, on="market_type", how="left")
    )

    print(f"Silhouette score: {sil:.4f}")
    print("Cluster sizes:")
    print(size_df.to_string(index=False))
    print("Cluster centroids:")
    print(centroid_df.to_string(index=False))
    print("Top 3 NTAs per cluster:")
    for market_type, group in clustered.groupby("market_type"):
        top3 = group.nlargest(3, "latent_demand_score")[
            [
                "nta_id",
                "demand_score",
                "latent_demand_score",
                "halal_supply_rate",
                "halal_cuisine_diversity",
            ]
        ]
        print(f"\n{market_type}")
        print(top3.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra_demand_cols = [c for c in ['demand_per_capita', 'population', 'demand_ci_lo', 'demand_ci_hi'] if c in clustered.columns]
    assignments = clustered[
        [
            'nta_id',
            'cluster_id',
            'market_type',
            'demand_score',
            'halal_supply_rate',
            'gap_score',
            'halal_cuisine_diversity',
        ]
        + extra_demand_cols
        + [
            c
            for c in [
                'cluster_confidence',
                'latent_demand_score',
                'halal_cuisine_diversity_norm',
                'total_restaurants',
                'halal_restaurants',
            ]
            if c in clustered.columns
        ]
    ].sort_values(['cluster_id', 'nta_id'])
    assignments.to_csv(OUT_DIR / "phase1_cluster_assignments.csv", index=False)
    centroid_df.to_csv(OUT_DIR / "phase1_cluster_centroids.csv", index=False)
    elbow_table.to_csv(OUT_DIR / "phase1_elbow_table.csv", index=False)


if __name__ == "__main__":
    main()
