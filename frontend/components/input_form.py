"""Input form — location and shortlist controls."""

from __future__ import annotations

import streamlit as st

from frontend.components._form_keys import FORM_KEYS

_ZONE_TYPE_HELP = {
    "All": "Show all zone types",
    "campus_walkshed": "10-min walk radius around universities — high student lunch demand",
    "lunch_corridor": "Dense daytime-worker catchments with quick-service lunch peaks",
    "transit_catchment": "Commuter-heavy areas around subway/rail hubs",
    "business_district": "Office-heavy cores with weekday lunch concentration",
    "nta_fallback": "Fallback NTA-based zones for areas not covered by curated business micro-zones",
}

_BOROUGH_HELP = (
    "Filter to a specific NYC borough, or leave as Any to search all five boroughs."
)


def render_input_form() -> dict[str, str | int]:
    """Render location filters and shortlist size."""
    zone_options = [
        "All",
        "campus_walkshed",
        "lunch_corridor",
        "transit_catchment",
        "business_district",
        "nta_fallback",
    ]
    zone_type = st.selectbox(
        "Preferred zone type",
        zone_options,
        key=FORM_KEYS["zone_type"],
        help="Filter recommendations to a specific micro-zone type. "
        + " | ".join(f"{k}: {v}" for k, v in _ZONE_TYPE_HELP.items() if k != "All"),
    )
    if zone_type and zone_type != "All":
        st.caption(_ZONE_TYPE_HELP.get(zone_type, ""))

    borough = st.selectbox(
        "Preferred borough",
        ["Any", "Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"],
        key=FORM_KEYS["borough"],
        help=_BOROUGH_HELP,
    )

    limit = st.slider(
        "Shortlist size",
        min_value=1,
        max_value=50,
        value=5,
        key=FORM_KEYS["limit"],
        help="Number of top-ranked micro-zones to display. Start with 5 for a focused shortlist.",
    )
    return {
        "zone_type": "" if zone_type == "All" else zone_type,
        "borough": "Any" if borough == "Any" else borough,
        "limit": limit,
    }
