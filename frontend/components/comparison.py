"""Comparison view — side-by-side neighborhood analysis."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from frontend.components.recommendation_card import _display_name, _build_radar_chart

def _delta_badge(val1, val2, label):
    delta = val1 - val2
    color = "#2a9d8f" if delta >= 0 else "#e63946"
    arrow = "↑" if delta >= 0 else "↓"
    return f"<span style='color: {color}; font-weight: bold;'>{arrow} {abs(delta):.2f}</span> {label}"

def _generate_narrative(row1, row2):
    n1 = _display_name(row1['nta_id'])
    n2 = _display_name(row2['nta_id'])
    
    better_demand = n1 if row1['demand_score'] > row2['demand_score'] else n2
    better_gap = n1 if row1['gap_score'] > row2['gap_score'] else n2
    
    return f"**{better_demand}** has stronger consumer interest, while **{better_gap}** offers a clearer market gap (less competition)."

def render_comparison_view(df: pd.DataFrame):
    if len(df) < 2:
        st.info("Add at least 2 neighborhoods to your shortlist to use the comparison view.")
        return

    st.markdown("### ⚖️ Side-by-Side Comparison")
    st.caption("Pick any two neighborhoods from your current shortlist to see how they stack up.")

    names = [_display_name(row['nta_id']) for _, row in df.iterrows()]
    id_map = { _display_name(row['nta_id']): row['nta_id'] for _, row in df.iterrows() }

    c1, c2 = st.columns(2)
    with c1:
        choice_a_name = st.selectbox("Select Neighborhood A", names, index=0)
    with c2:
        choice_b_name = st.selectbox("Select Neighborhood B", names, index=min(1, len(names)-1))

    row_a = df[df['nta_id'] == id_map[choice_a_name]].iloc[0]
    row_b = df[df['nta_id'] == id_map[choice_b_name]].iloc[0]

    st.divider()

    col_a, col_mid, col_b = st.columns([2, 1, 2])

    with col_a:
        st.markdown(f"#### {choice_a_name}")
        st.markdown(f"<div class='market-badge badge-{row_a['market_type'].lower().replace(' ', '-')}'>{row_a['market_type']}</div>", unsafe_allow_html=True)
        fig_a = _build_radar_chart(
            row_a['demand_score'], 
            row_a['gap_score'], 
            row_a['viability_score'], 
            row_a.get('high_risk_prob', 0.5),
            row_a.get('halal_demand_forecast_norm', 0.5)
        )
        st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False}, key=f"compare_radar_a_{row_a['nta_id']}")

    with col_mid:
        st.markdown("<h4 style='text-align: center; padding-top: 100px;'>VS</h4>", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"#### {choice_b_name}")
        st.markdown(f"<div class='market-badge badge-{row_b['market_type'].lower().replace(' ', '-')}'>{row_b['market_type']}</div>", unsafe_allow_html=True)
        fig_b = _build_radar_chart(
            row_b['demand_score'], 
            row_b['gap_score'], 
            row_b['viability_score'], 
            row_b.get('high_risk_prob', 0.5),
            row_b.get('halal_demand_forecast_norm', 0.5)
        )
        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False}, key=f"compare_radar_b_{row_b['nta_id']}")

    st.markdown("---")
    
    # Comparison Metrics Table-ish
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Overall Fit Score", f"{row_a['final_score']:.3f}", delta=f"{row_a['final_score']-row_b['final_score']:.3f}")
    with m2:
        st.metric("Halal Demand", f"{row_a['demand_score']:.3f}", delta=f"{row_a['demand_score']-row_b['demand_score']:.3f}")
    with m3:
        st.metric("Supply Gap", f"{row_a['gap_score']:.3f}", delta=f"{row_a['gap_score']-row_b['gap_score']:.3f}")

    st.markdown(f"> {_generate_narrative(row_a, row_b)}")
    
    # Risk Strip
    st.markdown("#### 🛡️ Risk Comparison")
    r_a, r_b = st.columns(2)
    r_a.info(f"**{choice_a_name}**: {row_a['risk_bucket']} Risk")
    r_b.info(f"**{choice_b_name}**: {row_b['risk_bucket']} Risk")
