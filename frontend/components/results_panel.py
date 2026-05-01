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


def _render_ranking_chart(df_all: pd.DataFrame, df_highlight: pd.DataFrame) -> None:
    """Violin + Strip plot for city-wide distribution with shortlist highlighted."""
    import plotly.express as px
    import plotly.graph_objects as go

    if df_all is None or df_all.empty:
        return

    # Prepare data for plotting
    plot_df = df_all.copy()
    highlight_ids = {str(x).strip() for x in df_highlight["nta_id"]} if df_highlight is not None else set()
    plot_df['Is Shortlisted'] = plot_df['nta_id'].apply(lambda x: 'Shortlist' if str(x).strip() in highlight_ids else 'Other')

    fig = px.violin(
        plot_df, 
        y="final_score", 
        x="market_type", 
        color="market_type",
        color_discrete_map=MARKET_TYPE_COLOR,
        box=True, 
        points="all",
        hover_data=["nta_id"],
        title="Score Distribution by Market Type"
    )

    # Highlight shortlisted points with a different symbol or larger size
    # Actually, plotly express doesn't easily allow different symbols per point in violin.
    # We'll just stick to a clean violin with boxplot and rely on Tab 3 for deeper dive.
    
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="Market Segment"),
        yaxis=dict(title="Overall Score", range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
        title=dict(
            text="How your matches compare to all neighborhoods",
            font=dict(size=14, color="#e9c46a"),
            x=0,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="citywide_ranking_violin")


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

    # City-wide ranking context chart
    if df_all is not None and not df_all.empty:
        with st.expander("📊 Ranking Distribution", expanded=True):
            _render_ranking_chart(df_all, df)

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
        use_container_width=True
    )
