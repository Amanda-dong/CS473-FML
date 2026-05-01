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
from frontend.pages.methodology import render_methodology_page

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
) -> pd.DataFrame:
    result = df.copy()
    if borough and borough != "Any":
        prefix = BOROUGH_PREFIX.get(borough, "")
        if prefix:
            result = result[result["nta_id"].str.startswith(prefix)]
    if market_type and market_type != "All":
        result = result[result["market_type"] == market_type]
    return result.sort_values("final_score", ascending=False).head(limit)


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
    )

    # Tabs
    tab_picks, tab_map, tab_method = st.tabs(
        ["🎯 Top Picks", "🗺️ Map View", "📖 Methodology"]
    )

    with tab_picks:
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NTAs Analyzed", len(df_all))
        col2.metric("Showing", len(filtered))
        col3.metric(
            "Top NTA",
            filtered.iloc[0]["nta_id"] if len(filtered) > 0 else "—",
        )
        col4.metric(
            "Avg Gap Score",
            f"{filtered['gap_score'].mean():.2f}" if len(filtered) > 0 else "—",
        )

        st.divider()
        render_results_panel(filtered)

    with tab_map:
        render_map_view(filtered)

    with tab_method:
        render_methodology_page()


if __name__ == "__main__":
    main()
