"""Input form used by the frontend workstream."""

import streamlit as st


def render_input_form() -> dict[str, str]:
    """Render the primary concept-input form and return user selections."""

    return {
        "zone_type": st.selectbox(
            "Zone type",
            ["campus_walkshed", "lunch_corridor", "transit_catchment", "business_district"],
        ),
        "borough": st.selectbox("Preferred borough", ["Any", "Brooklyn", "Manhattan", "Queens", "Bronx"]),
    }
