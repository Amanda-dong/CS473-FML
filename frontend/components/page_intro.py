from __future__ import annotations

import streamlit as st


def render_page_intro(title: str, body: str) -> None:
    """Render a compact, reusable page introduction."""
    st.markdown(f"### {title}")
    st.info(body)
