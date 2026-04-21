"""Input form — zone type, borough, and limit controls."""

from __future__ import annotations

import streamlit as st


def render_input_form() -> dict[str, str | int]:
    """Render zone/borough filters and return user selections."""
    zone_type = st.selectbox(
        "Zone type (optional filter)",
        ["All", "campus_walkshed", "lunch_corridor", "transit_catchment", "business_district"],
    )
    borough = st.selectbox(
        "Borough (optional filter)",
        ["Any", "Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"],
    )
    limit = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    return {
        "zone_type": "" if zone_type == "All" else zone_type,
        "borough": "Any" if borough == "Any" else borough,
        "limit": limit,
    }
