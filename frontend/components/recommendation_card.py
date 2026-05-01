"""Recommendation card — displays one NTA as a structured card."""

from __future__ import annotations

import streamlit as st

MARKET_TYPE_EMOJI = {
    "High Opportunity": "🔴",
    "Established Hub": "🔵",
    "Growing Market": "🟢",
    "Low Demand": "⚫",
}

RISK_COLOR = {
    "Low": "normal",
    "Medium": "off",
    "High": "inverse",
}

BOROUGH_NAME = {
    "BK": "Brooklyn",
    "QN": "Queens",
    "MN": "Manhattan",
    "BX": "Bronx",
    "SI": "Staten Island",
}


def _borough(nta_id: str) -> str:
    return BOROUGH_NAME.get(str(nta_id)[:2].upper(), "NYC")


def _fmt_pct(val) -> str:
    try:
        return f"{float(val) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_score(val) -> str:
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return "—"


def render_recommendation_card(row: dict, rank: int) -> None:
    nta_id = str(row.get("nta_id", ""))
    market_type = str(row.get("market_type", ""))
    final_score = row.get("final_score", 0.0)
    demand_score = row.get("demand_score", 0.0)
    gap_score = row.get("gap_score", 0.0)
    viability_score = row.get("viability_score", 0.5)
    halal_supply_rate = row.get("halal_supply_rate", 0.0)
    halal_cuisine_diversity = row.get("halal_cuisine_diversity", 0)
    risk_bucket = str(row.get("risk_bucket", "Unknown"))
    risk_confidence = str(row.get("risk_confidence", ""))
    high_risk_prob = row.get("high_risk_prob", 0.5)
    halal_demand_forecast = row.get("halal_demand_forecast", None)
    similar_ntas_raw = str(row.get("similar_ntas", "") or "")
    similar_ntas = [s.strip() for s in similar_ntas_raw.split(",") if s.strip()]

    emoji = MARKET_TYPE_EMOJI.get(market_type, "📍")

    with st.container(border=True):
        # Header
        col_rank, col_title, col_badge = st.columns([1, 4, 2])
        with col_rank:
            st.markdown(f"### #{rank}")
        with col_title:
            st.markdown(f"### {nta_id}")
            st.caption(f"{_borough(nta_id)}")
        with col_badge:
            st.markdown(f"**{emoji} {market_type}**")

        # Main scores
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Opportunity Score",
            _fmt_score(final_score),
            help="Weighted combination: 40% demand + 40% gap + 20% viability",
        )
        c2.metric(
            "Halal Discussion Signal",
            _fmt_pct(demand_score),
            help="Share of Yelp reviews mentioning halal (proxy, not true demand)",
        )
        c3.metric(
            "Opportunity Gap",
            _fmt_pct(gap_score),
            help="Demand proxy minus supply proxy (heuristic estimate)",
        )

        st.progress(float(final_score) if final_score else 0.0)

        # Supply info
        st.caption(
            f"Halal-relevant cuisine density: {_fmt_pct(halal_supply_rate)} of restaurants "
            f"| Cuisine types: {int(halal_cuisine_diversity) if halal_cuisine_diversity else 0}"
        )

        # Risk section
        with st.expander("Risk & Environment", expanded=False):
            rc1, rc2 = st.columns(2)
            rc1.metric(
                "Risk Level",
                risk_bucket,
                help="Rule-based risk index from inspection data",
            )
            rc2.metric(
                "Inspection Viability",
                _fmt_pct(viability_score),
            )
            if risk_confidence == "Low confidence":
                st.warning(
                    "⚠️ Low confidence: fewer than 10 inspection records for this NTA.",
                    icon="⚠️",
                )

        # Phase 3 insight
        with st.expander("Demand Insight (directional only)", expanded=False):
            if halal_demand_forecast is not None:
                try:
                    forecast_val = float(halal_demand_forecast)
                    st.metric(
                        "Halal Demand Forecast",
                        f"{forecast_val:.1%}",
                        help=(
                            "Ridge regression projection of 2023 halal review share. "
                            "R²=0.16 — treat as directional signal, not precise forecast."
                        ),
                    )
                except (TypeError, ValueError):
                    st.caption("Forecast unavailable for this NTA.")
            else:
                st.caption("Forecast unavailable for this NTA.")
            st.caption(
                "⚠️ This forecast does not beat the persistence baseline. "
                "Use for directional context only."
            )

        # Similar NTAs
        if similar_ntas:
            st.caption("Similar neighborhoods: " + " · ".join(similar_ntas))
