"""Streamlit entrypoint — NYC Halal Restaurant Opportunity Finder."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from frontend.components.comparison import render_comparison_view
from frontend.components.input_form import render_input_form
from frontend.components.map_view import render_map_view
from frontend.components.recommendation_card import _display_name, render_recommendation_card
from frontend.components.results_panel import render_analytics_view
from frontend.components.theme import inject_custom_theme
from frontend.review_evidence import load_labeled_reviews

DATA_PATH = _REPO_ROOT / "data" / "output"


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "final_recommendations.csv")


@st.cache_data(show_spinner=False)
def load_review_evidence_pool() -> pd.DataFrame | None:
    return load_labeled_reviews(_REPO_ROOT)


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
    return result if limit is None else result.head(limit)


def main() -> None:
    st.set_page_config(
        page_title="NYC Halal Opportunity Finder",
        page_icon="🕌",
        layout="wide",
    )
    inject_custom_theme()

    st.title("🕌 NYC Halal Restaurant Opportunity Finder")
    st.caption("Find promising NYC neighborhoods for a halal restaurant launch.")

    df_all = load_recommendations()

    with st.sidebar:
        st.header("Build Your Shortlist")
        st.caption("Filter by borough, market type, and risk tolerance.")
        form_state = render_input_form()
        st.divider()
        st.caption(f"Neighborhoods in model: **{len(df_all)}**")

    filtered_all = filter_recommendations(
        df_all,
        borough=form_state.get("borough"),
        market_type=form_state.get("market_type"),
        limit=None,
        risk_tolerance=form_state.get("risk_tolerance", "High"),
    )
    filtered = filtered_all.head(int(form_state.get("limit", 5)))
    review_pool = load_review_evidence_pool()

    tab1, tab2, tab3 = st.tabs(["🗺️ Explore", "⚖️ Compare", "📊 Analytics"])

    with tab1:
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Neighborhoods scored", len(df_all))
        kpi2.metric("Showing now", len(filtered))
        kpi3.metric("Risk filter", str(form_state.get("risk_tolerance", "Medium")))
        st.caption("Scores are relative rankings — not absolute measures.")

        render_map_view(filtered_all)
        st.divider()

        if filtered is None or filtered.empty:
            st.warning(
                "No neighborhoods match your filters. Try widening borough, risk, or market type."
            )
        else:
            top_row = filtered.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Top match", _display_name(str(top_row.get("nta_id", ""))))
            c2.metric("Best score", f"{float(top_row.get('final_score', 0.0)):.3f}")
            c3.metric("Top risk level", str(top_row.get("risk_bucket", "—")))


            st.subheader("Your Shortlist")
            col_left, col_right = st.columns(2)
            for i, (_, row) in enumerate(filtered.iterrows()):
                with col_left if i % 2 == 0 else col_right:
                    render_recommendation_card(
                        row.to_dict(),
                        rank=i + 1,
                        review_pool=review_pool,
                        repo_root=_REPO_ROOT,
                    )

            export_cols = [
                "nta_id", "market_type", "final_score", "demand_score",
                "gap_score", "viability_score", "risk_bucket", "similar_ntas",
            ]
            export_cols = [c for c in export_cols if c in filtered.columns]
            st.download_button(
                "📥 Export shortlist as CSV",
                data=filtered[export_cols].to_csv(index=False).encode("utf-8"),
                file_name="halal_recommendations.csv",
                mime="text/csv",
            )

    with tab2:
        st.markdown("Select two neighborhoods from your shortlist to compare them side by side.")
        if len(filtered) < 2:
            st.info(
                "Expand your shortlist (slider in sidebar) to at least 2 neighborhoods "
                "to use this view."
            )
        else:
            render_comparison_view(filtered)

    with tab3:
        render_analytics_view(df_all, filtered)
        st.divider()
        st.subheader("Methodology")
        from frontend.pages.methodology import render_methodology_page  # noqa: PLC0415
        render_methodology_page()


if __name__ == "__main__":
    main()
