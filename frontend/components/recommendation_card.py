"""Recommendation-card UI helpers."""

import streamlit as st


def render_recommendation_card(card: dict[str, object]) -> None:
    """Render a single placeholder recommendation card."""

    with st.container(border=True):
        st.markdown(f"### {card.get('zone_name', 'Placeholder Zone')}")
        st.write(f"Subtype: {card.get('concept_subtype', 'healthy_indian')}")
        st.write(f"Score: {card.get('opportunity_score', 0.0)}")
        st.write(card.get("healthy_gap_summary", "No gap summary yet."))
