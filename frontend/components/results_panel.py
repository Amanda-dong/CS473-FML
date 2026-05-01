"""Results panel — renders a list of recommendation cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components.recommendation_card import (
    _display_name,
    render_recommendation_card,
)

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


def render_results_panel(
    df: pd.DataFrame,
    *,
    repo_root=None,
    review_pool: pd.DataFrame | None = None,
) -> None:
    st.subheader("Best Matches")
    st.caption(
        "Start with the first card, then compare the rest by overall fit, risk, and similar nearby areas."
    )

    if df is None or df.empty:
        st.warning(
            "No neighborhoods match your current filters. Try widening the borough, allowing more risk, or choosing a broader market type."
        )
        return

    top_row = df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Top match", _display_name(str(top_row.get("nta_id", ""))))
    c2.metric("Best score", f"{float(top_row.get('final_score', 0.0)):.3f}")
    c3.metric("Top risk level", str(top_row.get("risk_bucket", "—")))
    st.caption("Higher scores rank first. Risk is shown separately in each card.")
    st.divider()

    for i, (_, row) in enumerate(df.iterrows()):
        render_recommendation_card(
            row.to_dict(),
            rank=i + 1,
            review_pool=review_pool,
            repo_root=repo_root,
        )

    # Export
    export_cols = [c for c in _EXPORT_COLUMNS if c in df.columns]
    csv_bytes = df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export as CSV",
        data=csv_bytes,
        file_name="halal_recommendations.csv",
        mime="text/csv",
    )
