"""Input form — borough, market type, and risk filters."""

from __future__ import annotations

import streamlit as st


def render_input_form() -> dict:
    borough = st.selectbox(
        "Borough",
        ["Any", "Brooklyn", "Queens", "Manhattan", "Bronx", "Staten Island"],
        help="Filter to a specific NYC borough.",
    )

    market_type = st.selectbox(
        "Market Type",
        ["All", "High Opportunity", "Established Hub", "Growing Market", "Low Demand"],
        help=(
            "High Opportunity: high halal demand, low supply. "
            "Established Hub: strong existing halal scene. "
            "Growing Market: moderate demand, little supply. "
            "Low Demand: limited halal activity."
        ),
    )

    limit = st.slider(
        "Results to show",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top-ranked NTAs to display.",
    )

    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["Low", "Medium", "High"],
        help="Choose how much neighborhood operating risk you are willing to accept.",
    )

    return {
        "borough": borough,
        "market_type": market_type,
        "limit": limit,
        "risk_tolerance": risk_tolerance,
    }
