from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.halal_risk import build_viability
from src.halal_similarity import build_similarity


OUT_DIR = ROOT / "data" / "output"
PHASE1 = OUT_DIR / "phase1_cluster_assignments.csv"
PHASE2 = OUT_DIR / "phase2_opportunity_scores.csv"


def main() -> None:
    df = pd.read_csv(PHASE1)
    risk = build_viability()
    df = df.merge(risk, on="nta_id", how="left")
    df["viability_score"] = df["viability_score"].fillna(0.5)
    df["risk_bucket"] = df["risk_bucket"].fillna("Unknown")

    df["final_score"] = (
        CFG.score_demand_weight * df["demand_score"] + CFG.score_gap_weight * df["gap_score"] + CFG.score_viability_weight * df["viability_score"]
    )

    feature_cols = ["demand_score", "halal_supply_rate", "gap_score", "viability_score"]
    df = build_similarity(df, feature_cols, top_n=CFG.similarity_top_n)

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    top10 = df.head(10)
    cluster_dist = (
        top10["market_type"]
        .value_counts()
        .rename_axis("market_type")
        .reset_index(name="count")
    )

    gap_rank = df["gap_score"].rank(ascending=False, method="average")
    final_rank = df["final_score"].rank(ascending=False, method="average")
    spearman_val = spearmanr(gap_rank, final_rank).correlation

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PHASE2, index=False)

    print("Cluster distribution of Top 10:")
    print(cluster_dist.to_string(index=False))
    print(
        f"\nSpearman correlation (gap_score rank vs final_score rank): {spearman_val:.4f}"
    )
    print("\nTop 5 NTA summary:")
    cols = [
        "rank",
        "nta_id",
        "market_type",
        "final_score",
        "demand_score",
        "gap_score",
        "viability_score",
        "risk_bucket",
        "similar_ntas",
    ]
    print(df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
