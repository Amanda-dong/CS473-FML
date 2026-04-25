"""Streamlit entrypoint — 3-tab shell for the NYC Healthy-Food White-Space Finder."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import frontend` when running `streamlit run frontend/app.py` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components._form_keys import FORM_DEFAULTS, FORM_KEYS
from frontend.components.data_freshness import render_data_freshness
from frontend.components.input_form import render_input_form
from frontend.components.map_view import render_map_view
from frontend.components.results_panel import render_results_panel
from frontend.components.scenario_panel import render_scenario_panel
from frontend.pages.methodology import render_methodology_page
from src.api.routers.recommendations import _get_zone_type_clusters, predict_cmf_sync
from src.schemas.requests import RecommendationRequest


# ---------------------------------------------------------------------------
# Cached inference wrappers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def _fetch_recs(
    concept_subtype: str,
    price_tier: str,
    borough: str | None,
    risk_tolerance: str,
    zone_type: str,
    limit: int,
) -> list[dict]:
    req = RecommendationRequest(
        concept_subtype=concept_subtype,
        price_tier=price_tier,
        borough=borough,
        risk_tolerance=risk_tolerance,
        zone_type=zone_type or "",
        limit=limit,
    )
    resp = predict_cmf_sync(req)
    return [
        r.model_dump() if hasattr(r, "model_dump") else dict(r)
        for r in resp.recommendations
    ]


@st.cache_resource(show_spinner=False)
def _fetch_clusters(
    concept_subtype: str, risk_tolerance: str, price_tier: str
) -> dict[str, str]:
    return _get_zone_type_clusters(concept_subtype, risk_tolerance, price_tier)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reset_filters() -> None:
    for key, default in FORM_DEFAULTS.items():
        widget_key = FORM_KEYS[key]
        st.session_state[widget_key] = default


def _render_zone_overview(recs: list[dict]) -> None:
    if not recs:
        return
    st.subheader("Zone Overview")
    df = pd.DataFrame(
        [
            {
                "Zone": r.get("zone_name", r.get("zone_label", "")),
                "Score": r.get("opportunity_score", 0.0) * 100,
                "Risk": r.get("survival_risk", 0.0) * 100,
                "Type": r.get("zone_type", ""),
            }
            for r in recs
        ]
    )
    fig = px.bar(
        df,
        x="Zone",
        y="Score",
        color="Type",
        title="Opportunity score by zone (%)",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_data_sources_tab() -> None:
    st.header("Data Sources & Freshness")
    render_data_freshness()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="NYC Healthy-Food White-Space", layout="wide")
    st.title("🥗 NYC Healthy-Food Restaurant White-Space Finder")

    # Sidebar
    with st.sidebar:
        st.header("Search Filters")
        form_state = render_input_form()
        scenario_state = render_scenario_panel()
        user_state = {**form_state, **scenario_state}
        st.divider()
        st.button(
            "🔄 Reset filters",
            on_click=_reset_filters,
            use_container_width=True,
        )

    tab_picks, tab_method, tab_data = st.tabs(
        ["🎯 Top Picks", "📖 Methodology", "📊 Data Sources"]
    )

    with tab_picks:
        concept = str(user_state.get("concept_subtype", "healthy_indian"))
        price = str(user_state.get("price_tier", "mid"))
        borough = user_state.get("borough") or None
        risk = str(user_state.get("risk_tolerance", "balanced"))
        zone_type = str(user_state.get("zone_type", "") or "")
        # "All" and "Any" are sentinel UI values meaning "no filter"
        if zone_type == "All":
            zone_type = ""
        if borough == "Any":
            borough = None
        limit = int(user_state.get("limit", 5))
        compare_mode = bool(user_state.get("compare_mode", False))
        compare_concept = user_state.get("compare_concept")

        # Header metrics row
        hm1, hm2 = st.columns(2)
        hm1.metric("NYC Zones Analyzed", "30")
        hm2.metric("Data Sources", "8")

        with st.spinner("Scoring zones..."):
            recs = _fetch_recs(concept, price, borough, risk, zone_type, limit)
            cluster_map = _fetch_clusters(concept, risk, price)

        _render_zone_overview(recs)
        render_map_view(recs)
        st.divider()

        if compare_mode and compare_concept and compare_concept != concept:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"A: {concept}")
                render_results_panel(
                    user_state,
                    recommendations=recs[:3],
                    cluster_map=cluster_map,
                )
            with col_b:
                st.subheader(f"B: {compare_concept}")
                recs_b = _fetch_recs(
                    compare_concept, price, borough, risk, zone_type, limit
                )
                cluster_map_b = _fetch_clusters(compare_concept, risk, price)
                render_results_panel(
                    {**user_state, "concept_subtype": compare_concept},
                    recommendations=recs_b[:3],
                    cluster_map=cluster_map_b,
                )
        else:
            render_results_panel(
                user_state, recommendations=recs, cluster_map=cluster_map
            )

    with tab_method:
        render_methodology_page()

    with tab_data:
        _render_data_sources_tab()


if __name__ == "__main__":
    main()
