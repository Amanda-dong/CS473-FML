"""Streamlit entrypoint — NYC Halal Restaurant Opportunity Finder."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from frontend.components.input_form import render_input_form
from frontend.components.map_view import render_map_view
from frontend.components.results_panel import render_results_panel

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = _REPO_ROOT / "data" / "output"


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "final_recommendations.csv")


@st.cache_data(show_spinner=False)
def load_phase1() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "phase1_cluster_assignments.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOROUGH_PREFIX = {
    "Brooklyn": "BK",
    "Queens": "QN",
    "Manhattan": "MN",
    "Bronx": "BX",
    "Staten Island": "SI",
}

MARKET_TYPE_COLOR = {
    "High Opportunity": "🔴",
    "Established Hub": "🔵",
    "Growing Market": "🟢",
    "Low Demand": "⚫",
}


def filter_recommendations(
    df: pd.DataFrame,
    borough: str | None,
    market_type: str | None,
    limit: int,
    risk_tolerance: str = "High",
) -> pd.DataFrame:
    result = df.copy()
    if borough and borough != "Any":
        prefix = BOROUGH_PREFIX.get(borough, "")
        if prefix:
            result = result[result["nta_id"].str.startswith(prefix)]
    if market_type and market_type != "All":
        result = result[result["market_type"] == market_type]
    if risk_tolerance == "Low":
        result = result[result["risk_bucket"] == "Low"]
    elif risk_tolerance == "Medium":
        result = result[result["risk_bucket"].isin(["Low", "Medium"])]
    result = result.sort_values("final_score", ascending=False)

    if result.empty:
        return result

    return result.head(limit)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="NYC Halal Opportunity Finder",
        page_icon="🕌",
        layout="wide",
    )
    st.title("🕌 NYC Halal Restaurant Opportunity Finder")
    st.caption("Find the best NYC neighborhoods to open a halal restaurant.")

    df_all = load_recommendations()

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        form_state = render_input_form()
        st.divider()
        st.caption(f"Total NTAs analyzed: **{len(df_all)}**")

    # Filter data
    filtered = filter_recommendations(
        df_all,
        borough=form_state.get("borough"),
        market_type=form_state.get("market_type"),
        limit=int(form_state.get("limit", 5)),
        risk_tolerance=form_state.get("risk_tolerance", "High"),
    )

    # Tabs
    (tab_overview,) = st.tabs(["📍 Overview"])

    with tab_overview:
        # Header metrics
        col1, col2 = st.columns(2)
        col1.metric("NTAs Analyzed", len(df_all))
        col2.metric("Showing", len(filtered))
        st.caption(
            "ℹ️ Signal strength labels (Very Strong / Moderate / Weak / Low) are relative rankings across 144 NYC neighborhoods, not absolute percentages."
        )

        st.divider()
        st.subheader("Neighborhood Map")
        render_map_view(filtered)
        st.divider()
        st.subheader("Recommendations")
        render_results_panel(filtered)


if __name__ == "__main__":
    main()
