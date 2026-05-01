"""Input form — borough, market type, and risk filters."""

from __future__ import annotations

import streamlit as st


def render_input_form() -> dict:
    borough = st.selectbox(
        "Borough",
        ["Any", "Brooklyn", "Queens", "Manhattan", "Bronx", "Staten Island"],
        help="Use Any to scan the full city, or pick one borough you want to focus on.",
    )

    market_type = st.selectbox(
        "Market type",
        ["All", "High Opportunity", "Established Hub", "Growing Market", "Low Demand"],
        help=(
            "High Opportunity means strong local interest with less current halal supply. "
            "Growing Market is a middle ground. "
            "Established Hub means more existing halal competition."
        ),
    )

    limit = st.slider(
        "Shortlist size",
        min_value=1,
        max_value=20,
        value=5,
        help="Choose how many neighborhoods you want to compare at once.",
    )

    risk_tolerance = st.selectbox(
        "Max risk allowed",
        ["Low", "Medium", "High"],
        index=2,
        help="Low shows only lower-risk neighborhoods. Medium adds moderate-risk options. High shows the full list.",
    )

    return {
        "borough": borough,
        "market_type": market_type,
        "limit": limit,
        "risk_tolerance": risk_tolerance,
    }
