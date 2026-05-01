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
    """Horizontal bar chart — all NTAs in grey, highlighted shortlist in color."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    if df_all is None or df_all.empty:
        return

    plot_df = df_all.sort_values("final_score", ascending=True).copy()
    highlight_ids = (
        {str(x).strip() for x in df_highlight["nta_id"]}
        if df_highlight is not None
        else set()
    )

    def _is_highlighted(row):
        return str(row.get("nta_id", "")).strip() in highlight_ids

    colors = [
        MARKET_TYPE_COLOR.get(str(row.get("market_type", "")).strip(), "#adb5bd")
        if _is_highlighted(row)
        else "rgba(180,180,180,0.3)"
        for _, row in plot_df.iterrows()
    ]
    labels = [
        _display_name(str(nta).strip()) if str(nta).strip() in highlight_ids else ""
        for nta in plot_df["nta_id"]
    ]

    fig = go.Figure(
        go.Bar(
            x=plot_df["final_score"].tolist(),
            y=list(range(len(plot_df))),
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{customdata[0]}</b><br>Score: %{x:.3f}<br>Type: %{customdata[1]}<extra></extra>",
            customdata=list(
                zip(
                    plot_df["nta_id"].astype(str).tolist(),
                    plot_df["market_type"].astype(str).tolist(),
                )
            ),
        )
    )
    fig.update_layout(
        height=max(180, min(len(plot_df) * 5, 340)),
        margin=dict(l=0, r=120, t=8, b=8),
        xaxis=dict(title="Overall Score", range=[0, 1.05]),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text="Your shortlist vs all scored neighborhoods",
            font=dict(size=12),
            x=0,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_results_panel(
    df: pd.DataFrame,
    *,
    repo_root=None,
    review_pool: pd.DataFrame | None = None,
    df_all: pd.DataFrame | None = None,
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

    # City-wide ranking context chart
    if df_all is not None and not df_all.empty:
        with st.expander("📊 Where your shortlist ranks city-wide", expanded=True):
            _render_ranking_chart(df_all, df)
            st.caption(
                "Colored bars = your current shortlist (by market type). "
                "Grey bars = all other scored neighborhoods."
            )

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
