"""Results panel — renders ranking bar chart + recommendation cards."""

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

MARKET_TYPE_COLOR = {
    "High Opportunity": "#e63946",
    "Established Hub": "#457b9d",
    "Growing Market": "#2a9d8f",
    "Low Demand": "#adb5bd",
}


def render_analytics_view(df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    """Rich analytics for Tab 3."""
    import plotly.express as px
    
    st.markdown("### 🔍 Market Analysis Deep-Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Score Distribution by Market Type")
        fig_box = px.box(
            df_all, 
            x="market_type", 
            y="final_score", 
            color="market_type",
            color_discrete_map=MARKET_TYPE_COLOR,
            points="all",
            title="Where does your shortlist sit?"
        )
        fig_box.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#fafafa"}
        )
        st.plotly_chart(fig_box, use_container_width=True, key="analytics_box_market")

    with col2:
        st.markdown("#### Demand vs. Supply Gap")
        fig_scatter = px.scatter(
            df_all,
            x="demand_score",
            y="gap_score",
            color="market_type",
            size="final_score",
            hover_name="nta_id",
            color_discrete_map=MARKET_TYPE_COLOR,
            title="Strategic Positioning"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#fafafa"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="analytics_scatter_gap")

    st.divider()
    
    st.markdown("#### 📋 Full Comparison Table")
    st.caption("Sort and filter the entire dataset used for this model.")
    
    st.dataframe(
        df_all[[
            "nta_id", "market_type", "final_score", 
            "demand_score", "gap_score", "viability_score", "risk_bucket"
        ]].sort_values("final_score", ascending=False),
        use_container_width=True,
        hide_index=True
    )


def render_results_panel(
    df: pd.DataFrame,
    *,
    repo_root=None,
    review_pool: pd.DataFrame | None = None,
    df_all: pd.DataFrame | None = None,
) -> None:
    st.subheader("Best Matches")
    
    with st.expander("🧪 Formula Sandbox (Weights)", expanded=False):
        st.markdown("""
        The **Overall Fit Score** is currently calculated as:
        - **40%** Halal Demand Signal
        - **40%** Supply Gap (Unmet Demand)
        - **20%** Neighborhood Viability (Operating Safety)
        """)
        st.info("Custom weight adjustment is coming in Phase 4.")

    if df is None or df.empty:
        st.warning(
            "No neighborhoods match your current filters."
        )
        return

    top_row = df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Top match", _display_name(str(top_row.get("nta_id", ""))))
    c2.metric("Best score", f"{float(top_row.get('final_score', 0.0)):.3f}")
    c3.metric("Top risk level", str(top_row.get("risk_bucket", "—")))

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
        "📥 Export Shortlist as CSV",
        data=csv_bytes,
        file_name="halal_shortlist.csv",
        mime="text/csv",
        use_container_width=True,
        key="export_shortlist_btn"
    )
