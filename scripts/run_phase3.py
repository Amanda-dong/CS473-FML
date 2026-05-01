from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.halal_forecast import build_entry_forecast, build_forecast
from src.halal_risk import build_gmm_risk
from src.halal_spatial import build_lisa
from src.utils import minmax as _minmax


OUT_DIR = ROOT / "data" / "output"
PHASE2 = OUT_DIR / "phase2_opportunity_scores.csv"
RISK_OUT = OUT_DIR / "phase3_risk_scores.csv"
FINAL_OUT = OUT_DIR / "final_recommendations.csv"


def main() -> None:
    risk_df, risk_diag = build_gmm_risk()
    forecast_df, forecast_diag = build_forecast()
    entry_df, entry_diag = build_entry_forecast()

    print("GMM cluster means table:")
    print(risk_diag["cluster_means"].to_string(index=False))
    print("\nGMM component inspection coverage:")
    print(risk_diag["component_inspections"].to_string(index=False))
    print(f"\nGMM silhouette score: {risk_diag['silhouette']:.4f}")
    print("GMM BIC table:")
    print(risk_diag["bic_table"].to_string(index=False))

    print(
        f"\nRidge R² (in-sample): {forecast_diag['r2_insample']:.4f}"
    )
    print(f"Ridge persistence baseline R²: {forecast_diag['baseline_r2']:.4f}")
    print("\nRidge feature coefficients:")
    print(forecast_diag["coefficients"].to_string(index=False))
    print("\nRidge ablation table:")
    print(
        forecast_diag["ablation"].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print("\nPredicted vs actual halal_related_share — Top 5 actual:")
    print(forecast_diag["top_actual"].to_string(index=False))
    print("\nPredicted vs actual halal_related_share — Bottom 5 actual:")
    print(forecast_diag["bottom_actual"].to_string(index=False))

    print(
        f"\nEntry Ridge R² (in-sample): {entry_diag.get('r2_insample', entry_diag.get('r2_mean', 0.0)):.4f}"
    )
    print(f"Entry Ridge persistence baseline R²: {entry_diag.get('baseline_r2', 0.0):.4f}")
    print("\nEntry Ridge feature coefficients:")
    print(entry_diag["coefficients"].to_string(index=False))
    print("\nEntry Ridge ablation table:")
    print(
        entry_diag["ablation"].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nPredicted vs actual new halal count — Top 5 actual:")
    print(entry_diag["top_actual"].to_string(index=False))
    print("\nPredicted vs actual new halal count — Bottom 5 actual:")
    print(entry_diag["bottom_actual"].to_string(index=False))

    phase2 = pd.read_csv(PHASE2).copy()
    # --- LISA spatial opportunity detection ---
    try:
        lisa_df = build_lisa(phase2[["nta_id", "gap_score"]].copy())
        lisa_available = True
    except Exception as e:
        print(f"LISA skipped: {e}")
        lisa_df = None
        lisa_available = False

    phase2 = phase2.drop(
        columns=[
            c
            for c in [
                "high_risk_prob",
                "risk_bucket",
                "risk_confidence",
                "halal_demand_forecast",
                "halal_demand_forecast_norm",
                "new_halal_entry_forecast",
                "final_score_adjusted",
            ]
            if c in phase2.columns
        ],
        errors="ignore",
    )

    final = (
        phase2.merge(risk_df, on="nta_id", how="left")
        .merge(forecast_df, on="nta_id", how="left")
        .merge(entry_df, on="nta_id", how="left")
    )
    if lisa_available and lisa_df is not None:
        final = final.merge(lisa_df, on="nta_id", how="left")
        for col in ["moran_ii", "moran_p", "moran_q", "lisa_opportunity"]:
            if col in final.columns:
                if col == "lisa_opportunity":
                    final[col] = final[col].fillna(False)
                elif col == "moran_ii":
                    final[col] = final[col].fillna(0.0)
                elif col == "moran_p":
                    final[col] = final[col].fillna(1.0)
                else:
                    final[col] = final[col].fillna("LL")

    final["high_risk_prob"] = final["high_risk_prob"].fillna(0.5)
    median_forecast = (
        float(forecast_df["halal_demand_forecast"].median())
        if len(forecast_df)
        else 0.5
    )
    final["halal_demand_forecast"] = final["halal_demand_forecast"].fillna(
        median_forecast
    )
    median_entry = (
        float(entry_df["new_halal_entry_forecast"].median()) if len(entry_df) else 0.0
    )
    final["new_halal_entry_forecast"] = final["new_halal_entry_forecast"].fillna(
        median_entry
    )
    final["risk_bucket"] = final["risk_bucket"].fillna("Unknown")
    final["risk_confidence"] = final["risk_confidence"].fillna("Low confidence")
    final["halal_demand_forecast_norm"] = _minmax(final["halal_demand_forecast"])
    final["final_score_adjusted"] = (
        final["final_score"]
        * (1 - CFG.risk_penalty * final["high_risk_prob"])
        * (1 + CFG.forecast_boost * final["halal_demand_forecast_norm"])
    )
    if lisa_available and "lisa_opportunity" in final.columns:
        final["final_score_adjusted"] = final["final_score_adjusted"] * (
            1 + 0.08 * final["lisa_opportunity"].fillna(False).astype(float)
        )

    final = final.sort_values("final_score_adjusted", ascending=False).reset_index(
        drop=True
    )
    final["rank"] = range(1, len(final) + 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        risk_df.merge(forecast_df, on="nta_id", how="outer")
        .merge(entry_df, on="nta_id", how="outer")
        .to_csv(RISK_OUT, index=False)
    )
    final.to_csv(FINAL_OUT, index=False)

    print("\nUpdated Top 5:")
    cols = [
        "rank",
        "nta_id",
        "market_type",
        "final_score_adjusted",
        "high_risk_prob",
        "risk_bucket",
        "risk_confidence",
        "halal_demand_forecast",
        "new_halal_entry_forecast",
    ]
    print(final[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
