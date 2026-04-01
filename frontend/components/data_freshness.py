"""Data freshness helpers for the frontend."""

import streamlit as st


def render_data_freshness(note: str) -> None:
    """Render a compact freshness note."""

    st.caption(f"Data freshness: {note}")
