"""Results panel — renders a list of recommendation cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components.recommendation_card import render_recommendation_card

_EXPORT_COLUMNS = [
    "nta_id",
    "market_type",
    "final_score",
    "demand_score",
    "gap_score",
    "viability_score",
    "risk_bucket",
    "similar_ntas",
]


def _render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    top_market = df["market_type"].mode().iloc[0] if not df.empty else "—"
    avg_gap = df["gap_score"].mean()
    low_risk_count = (df["risk_bucket"] == "Low").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Top market type", top_market)
    c2.metric("Avg gap score", f"{avg_gap:.2f}")
    c3.metric("Low-risk NTAs", f"{low_risk_count} of {len(df)}")


def render_results_panel(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No NTAs match the current filters. Try broadening your search.")
        return

    _render_summary(df)
    st.divider()

    for i, (_, row) in enumerate(df.iterrows()):
        render_recommendation_card(row.to_dict(), rank=i + 1)

    # Export
    export_cols = [c for c in _EXPORT_COLUMNS if c in df.columns]
    csv_bytes = df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export as CSV",
        data=csv_bytes,
        file_name="halal_recommendations.csv",
        mime="text/csv",
    )
