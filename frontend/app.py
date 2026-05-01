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


BOROUGH_PREFIX = {
    "Brooklyn": "BK",
    "Queens": "QN",
    "Manhattan": "MN",
    "Bronx": "BX",
    "Staten Island": "SI",
}


def filter_recommendations(
    df: pd.DataFrame,
    borough: str | None,
    market_type: str | None,
    limit: int | None,
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

    if limit is None:
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
    st.caption("Find promising NYC neighborhoods for a halal restaurant launch.")

    df_all = load_recommendations()

    # Sidebar filters
    with st.sidebar:
        st.header("Build Your Shortlist")
        st.caption(
            "Choose the borough, market type, and risk level that fit your halal restaurant plan."
        )
        form_state = render_input_form()
        st.divider()
        st.caption(f"Neighborhoods in model: **{len(df_all)}**")

    # Filter data
    filtered_all = filter_recommendations(
        df_all,
        borough=form_state.get("borough"),
        market_type=form_state.get("market_type"),
        limit=None,
        risk_tolerance=form_state.get("risk_tolerance", "High"),
    )
    filtered = filtered_all.head(int(form_state.get("limit", 5)))

    st.subheader("Welcome")
    st.caption(
        "This dashboard helps you compare NYC neighborhoods for a halal restaurant launch. Start with the best matches below, then use the map and neighborhood details to narrow your options."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Neighborhoods scored", len(df_all))
    col2.metric("Showing now", len(filtered))
    col3.metric(
        "Risk filter",
        str(form_state.get("risk_tolerance", "Medium")),
    )
    st.caption(
        "Labels like Very Strong or Moderate are simple relative rankings across the neighborhoods in this model."
    )

    st.divider()
    render_results_panel(filtered)
    st.divider()
    render_map_view(filtered_all)


if __name__ == "__main__":
    main()
