"""Scenario controls for the recommendation prototype."""

import streamlit as st


def render_scenario_panel() -> dict[str, str]:
    """Render a minimal control panel for scenario testing."""

    return {
        "concept_subtype": st.selectbox(
            "Healthy concept subtype",
            ["healthy_indian", "mediterranean_bowls", "salad_bowls", "vegan_grab_and_go"],
        ),
        "price_tier": st.selectbox("Price tier", ["budget", "mid", "premium"]),
        "risk_tolerance": st.selectbox("Risk tolerance", ["conservative", "balanced", "aggressive"]),
    }
