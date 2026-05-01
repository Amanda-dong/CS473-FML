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
from frontend.components.theme import inject_custom_theme
from frontend.review_evidence import load_labeled_reviews

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = _REPO_ROOT / "data" / "output"


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "final_recommendations.csv")


@st.cache_data(show_spinner=False)
def load_review_evidence_pool() -> pd.DataFrame | None:
    """Yelp reviews with Gemini halal labels — used for qualitative evidence per NTA."""
    return load_labeled_reviews(_REPO_ROOT)


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
    result = result.sort_values("final_score_adjusted", ascending=False)

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
    inject_custom_theme()

    df_all = load_recommendations()

    # Sidebar filters
    with st.sidebar:
        st.markdown("<h2 style='color: #e9c46a;'>🕌 Halal Scout</h2>", unsafe_allow_html=True)
        st.caption(
            "Build your shortlist by choosing the borough, market type, and risk level."
        )
        form_state = render_input_form()
        
        if st.button("🔄 Quick Reset", use_container_width=True):
            st.rerun()
            
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
    selected_market = form_state.get("market_type")
    selected_risk = form_state.get("risk_tolerance", "High")
    if (
        filtered_all.empty
        and selected_market not in (None, "All")
        and selected_risk != "High"
    ):
        filtered_all = filter_recommendations(
            df_all,
            borough=form_state.get("borough"),
            market_type=selected_market,
            limit=None,
            risk_tolerance="High",
        )
        if not filtered_all.empty:
            st.info(
                f"No rows matched `{selected_market}` under risk `{selected_risk}`. "
                "Showing available rows with `High` risk."
            )
    limit_val = int(form_state.get("limit", 5))
    filtered = filtered_all.head(limit_val)

    # Tabs
    tab_map, tab_compare, tab_analytics = st.tabs([
        "📍 Map & Shortlist", 
        "⚖️ Compare", 
        "📊 Analytics"
    ])

    with tab_map:
        st.markdown("### 🗺️ Opportunity Map")
        col_map, col_summary = st.columns([2, 1])
        
        with col_map:
            render_map_view(filtered_all)
            
        with col_summary:
            st.markdown("#### 🏆 Top 3 Summary")
            top_3 = filtered.head(3)
            if top_3.empty:
                st.info("No matches found.")
            for _, row in top_3.iterrows():
                with st.container():
                    st.markdown(f"**{row['nta_id']}**")
                    score_val = row.get('final_score_adjusted', row.get('final_score', 0.0))
                    c1, c2 = st.columns(2)
                    c1.metric("Score", f"{score_val:.3f}")
                    c2.markdown(f"<div class='market-badge badge-{row['market_type'].lower().replace(' ', '-')}'>{row['market_type']}</div>", unsafe_allow_html=True)
            st.caption("Scroll down for full details and review evidence.")

        st.divider()
        review_pool = load_review_evidence_pool()
        render_results_panel(
            filtered,
            repo_root=_REPO_ROOT,
            review_pool=review_pool,
            df_all=df_all,
        )

    with tab_compare:
        from frontend.components.comparison import render_comparison_view
        render_comparison_view(filtered)

    with tab_analytics:
        st.subheader("📊 Market Analytics")
        # results_panel will handle the rich analytics in step 5
        from frontend.components.results_panel import render_analytics_view
        render_analytics_view(filtered_all, filtered)


if __name__ == "__main__":
    main()
