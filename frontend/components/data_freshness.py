"""Data freshness helpers for the frontend."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

_SOURCES: list[tuple[str, str, str, str | None]] = [
    # (tier, name, freshness_note, parquet_check)
    (
        "T1",
        "NYC DOHMH Inspections",
        "Refreshed weekly from NYC Open Data",
        "data/processed/inspections.parquet",
    ),
    (
        "T1",
        "NYC DOB Permits",
        "Refreshed weekly from NYC Open Data",
        "data/processed/permits.parquet",
    ),
    (
        "T1",
        "Citi Bike trips",
        "Monthly S3 dumps (202603 snapshot)",
        "data/processed/citibike.parquet",
    ),
    ("T1", "U.S. Census ACS 5-year", "Annual release (2023 vintage)", None),
    (
        "T1",
        "NYC NTA boundaries",
        "Static shapefile (2020 tabulation)",
        "data/processed/boundaries.parquet",
    ),
    (
        "T2",
        "Yelp Fusion API",
        "Pulled on-demand; 24h cache",
        "data/processed/yelp.parquet",
    ),
    ("T2", "Inside Airbnb", "Quarterly scrape snapshot", None),
    (
        "T2",
        "NYC 311 Complaints",
        "Monthly export from NYC Open Data",
        "data/processed/complaints_311.parquet",
    ),
]

_PARQUET_CHECKS = {
    "NYC DOHMH Inspections": "data/processed/inspections.parquet",
    "NYC DOB Permits": "data/processed/permits.parquet",
    "Citi Bike trips": "data/processed/citibike.parquet",
    "NYC NTA boundaries": "data/processed/boundaries.parquet",
    "Yelp Fusion API": "data/processed/yelp.parquet",
    "NYC 311 Complaints": "data/processed/complaints_311.parquet",
    "U.S. Census ACS 5-year": None,
    "Inside Airbnb": None,
}


def render_data_freshness(note: str = "") -> None:
    """Render per-source freshness with live availability status."""
    st.markdown(f"**Last page load:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("If a source is stale or missing, treat the shortlist as directional.")
    st.divider()

    available = 0
    for tier, name, freshness_note, parquet_path in _SOURCES:
        col_status, col_badge, col_body = st.columns([0.5, 0.8, 5])
        parquet_file = _PARQUET_CHECKS.get(name)
        loaded = parquet_file is not None and Path(parquet_file).exists()
        if parquet_file is None:
            status_icon = "✅"
            available += 1
        elif loaded:
            status_icon = "✅"
            available += 1
        else:
            status_icon = "⚠️"
        with col_status:
            st.markdown(status_icon)
        with col_badge:
            st.markdown(f"`[{tier}]`")
        with col_body:
            st.markdown(f"**{name}** — {freshness_note}")

    st.divider()
    st.caption(f"{available}/{len(_SOURCES)} sources available in processed cache.")
    if note:
        st.caption(note)
