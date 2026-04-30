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
from frontend.components.page_intro import render_page_intro
from frontend.components.results_panel import (
    render_results_panel,
    render_top_match_panel,
)
from frontend.components.scenario_panel import render_scenario_panel
from src.api.routers.recommendations import _get_zone_type_clusters, predict_cmf_sync
from src.schemas.requests import RecommendationRequest


SUBTYPE_LABELS = {
    "halal": "Halal Fast Casual",
    "salad_bowls": "Salad Bowls",
    "mediterranean_bowls": "Mediterranean Bowls",
    "healthy_indian": "Healthy Indian",
    "vegan_grab_and_go": "Vegan Grab-and-Go",
    "protein_forward_lunch": "Protein-Forward Lunch",
    "mexican": "Mexican",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean",
    "thai": "Thai",
    "italian": "Italian",
    "greek": "Greek",
    "middle_eastern": "Middle Eastern",
    "caribbean": "Caribbean",
    "ethiopian": "Ethiopian",
    "west_african": "West African",
    "american_comfort": "American Comfort",
    "burgers": "Burgers",
    "pizza": "Pizza",
    "seafood": "Seafood",
    "ramen": "Ramen",
    "dim_sum": "Dim Sum",
    "bakery_cafe": "Bakery & Cafe",
    "smoothie_juice": "Smoothie & Juice",
}


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
        if widget_key in st.session_state:
            st.session_state[widget_key] = default
    st.session_state["query_submitted"] = False
    st.session_state["submitted_query"] = None


def _submit_query(user_state: dict) -> None:
    st.session_state["query_submitted"] = True
    st.session_state["submitted_query"] = dict(user_state)


def _render_zone_overview(recs: list[dict]) -> None:
    if not recs:
        return
    st.subheader("Zone Overview")
    st.caption("Scan the table, then use the chart to spot the strongest zones.")
    df = pd.DataFrame(
        [
            {
                "Zone": r.get("zone_name", r.get("zone_label", "")),
                "Score": f"{float(r.get('opportunity_score', 0.0) or 0.0) * 100:.0f}%",
                "Confidence": str(r.get("confidence_bucket", "—")).title(),
                "Risk": f"{float(r.get('survival_risk', 0.0) or 0.0) * 100:.0f}%",
                "Type": r.get("zone_type", ""),
                "_Score_Num": float(r.get("opportunity_score", 0.0) or 0.0) * 100,
            }
            for r in recs
        ]
    )

    st.dataframe(
        df.drop(columns=["_Score_Num"]), use_container_width=True, hide_index=True
    )

    fig = px.bar(
        df,
        x="Zone",
        y="_Score_Num",
        color="Type",
        hover_data={
            "_Score_Num": False,
            "Score": True,
            "Confidence": True,
            "Risk": True,
            "Type": True,
        },
        title="Opportunity score by zone (%)",
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Score"
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_data_sources_tab() -> None:
    st.header("Data Sources & Freshness")
    render_page_intro(
        "What this section shows",
        "Check data freshness before you trust a shortlist.",
    )
    render_data_freshness()


def _render_current_query(active_query: dict) -> None:
    """Show the active submitted query so users know what drove the results."""
    concept = str(active_query.get("concept_subtype", "halal"))
    concept_label = SUBTYPE_LABELS.get(concept, concept.replace("_", " ").title())
    borough = str(active_query.get("borough", "Any") or "Any")
    zone_type = str(active_query.get("zone_type", "All") or "All")
    price = str(active_query.get("price_tier", "mid")).title()
    risk = str(active_query.get("risk_tolerance", "balanced")).title()
    limit = int(active_query.get("limit", 5))
    concept_mode = str(active_query.get("concept_mode", "Use structured controls"))
    concept_description = str(active_query.get("concept_description", "") or "").strip()
    use_nlp_suggestions = bool(active_query.get("use_nlp_suggestions", False))

    st.subheader("Current Query")
    st.caption("These settings produced the shortlist below.")

    q1, q2, q3 = st.columns(3)
    q1.metric("Concept", concept_label)
    q2.metric("Price tier", price)
    q3.metric("Risk tolerance", risk)

    st.markdown(
        f"**Borough:** {borough}  |  **Zone type:** {zone_type.replace('_', ' ').title() if zone_type != 'All' else 'All'}  |  **Shortlist size:** {limit}"
    )
    mode_label = concept_mode
    if concept_mode == "Describe my halal concept" and concept_description:
        mode_label += (
            " (NLP suggestions on)" if use_nlp_suggestions else " (manual price/risk)"
        )
    st.markdown(f"**Input mode:** {mode_label}")
    if concept_description:
        st.markdown(f"**Merchant description:** {concept_description}")
    st.divider()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="NYC Healthy-Food White-Space", layout="wide")
    st.title("🕌 NYC Halal Restaurant White-Space Finder")

    # Sidebar
    with st.sidebar:
        st.header("Plan Your Halal Concept")
        st.caption("Set your concept and filters, then run the search.")
        with st.form("halal_query_form", clear_on_submit=False):
            form_state = render_input_form()
            scenario_state = render_scenario_panel()
            user_state = {**form_state, **scenario_state}
            run_query = st.form_submit_button(
                "Find Best Matches",
                use_container_width=True,
                type="primary",
            )
            if run_query:
                _submit_query(user_state)
        st.divider()
        st.button(
            "🔄 Reset filters",
            on_click=_reset_filters,
            use_container_width=True,
        )

    tab_picks, tab_data = st.tabs(["🎯 Top Picks", "📊 Data Sources"])

    with tab_picks:
        render_page_intro(
            "How to use this page",
            "Describe your halal concept in the left sidebar, click **Find Best Matches**, then review the shortlist, map, and top recommendation.",
        )
        active_query = st.session_state.get("submitted_query") or {}
        query_submitted = bool(st.session_state.get("query_submitted", False))

        concept = str(active_query.get("concept_subtype", "halal"))
        price = str(active_query.get("price_tier", "mid"))
        borough = active_query.get("borough") or None
        risk = str(active_query.get("risk_tolerance", "balanced"))
        zone_type = str(active_query.get("zone_type", "") or "")
        # "All" and "Any" are sentinel UI values meaning "no filter"
        if zone_type == "All":
            zone_type = ""
        if borough == "Any":
            borough = None
        limit = int(active_query.get("limit", 5))
        compare_mode = bool(active_query.get("compare_mode", False))
        compare_concept = active_query.get("compare_concept")
        # Header metrics row
        hm1, hm2 = st.columns(2)
        hm1.metric("NYC zones analyzed", "30")
        hm2.metric("Data sources", "8")

        if not query_submitted:
            st.info(
                "No results yet. Use the left sidebar, then click `Find Best Matches`."
            )
        else:
            _render_current_query(active_query)
            with st.spinner("Scoring zones..."):
                recs = _fetch_recs(concept, price, borough, risk, zone_type, limit)
                cluster_map = _fetch_clusters(concept, risk, price)

            featured_zone_id = render_top_match_panel(
                active_query,
                recommendation=recs[0] if recs else None,
                cluster_map=cluster_map,
            )
            st.divider()
            _render_zone_overview(recs)
            render_map_view(recs)
            st.divider()

            if compare_mode and compare_concept and compare_concept != concept:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader(f"A: {concept}")
                    render_results_panel(
                        active_query,
                        recommendations=recs[:3],
                        cluster_map=cluster_map,
                        featured_zone_id=featured_zone_id,
                    )
                with col_b:
                    st.subheader(f"B: {compare_concept}")
                    recs_b = _fetch_recs(
                        compare_concept, price, borough, risk, zone_type, limit
                    )
                    cluster_map_b = _fetch_clusters(compare_concept, risk, price)
                    render_results_panel(
                        {**active_query, "concept_subtype": compare_concept},
                        recommendations=recs_b[:3],
                        cluster_map=cluster_map_b,
                    )
            else:
                render_results_panel(
                    active_query,
                    recommendations=recs,
                    cluster_map=cluster_map,
                    featured_zone_id=featured_zone_id,
                )

    with tab_data:
        _render_data_sources_tab()


if __name__ == "__main__":
    main()
