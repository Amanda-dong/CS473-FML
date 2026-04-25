from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.models.explainability import FEATURE_DISPLAY_NAMES


_ZONE_TYPE_BADGE = {
    "campus_walkshed": "🎓 Campus",
    "lunch_corridor": "🥗 Lunch Corridor",
    "transit_catchment": "🚇 Transit",
    "business_district": "💼 Business District",
}
_CLUSTER_BADGE = {
    "emerging": "🌱 Emerging",
    "gentrifying": "📈 Gentrifying",
    "stable": "🏛️ Stable",
    "declining": "📉 Declining",
}
_CONFIDENCE_BADGE = {
    "high": "🟢 High confidence",
    "medium": "🟡 Medium confidence",
    "low": "🔴 Low confidence",
}
_SCORE_COLOR = {
    "high": "normal",
    "medium": "off",
    "low": "inverse",
}


def _build_gap_summary(card: dict, cluster: str) -> str:
    contribs = card.get("feature_contributions") or {}
    if contribs:
        try:
            top_key = max(contribs, key=lambda k: contribs[k])
        except ValueError:
            top_key = ""
        if top_key:
            top_name = FEATURE_DISPLAY_NAMES.get(
                top_key, top_key.replace("_", " ").title()
            )
            cluster_part = f" Cluster: {cluster}." if cluster else ""
            return f"Driven by strong {top_name}.{cluster_part}"
    return "Opportunity signal derived from heuristic scoring."


def _render_driver_chart(feature_contributions: dict) -> None:
    if not feature_contributions:
        st.info("Score breakdown unavailable.")
        return
    items = sorted(
        feature_contributions.items(), key=lambda kv: abs(float(kv[1])), reverse=True
    )[:8]
    labels = [
        FEATURE_DISPLAY_NAMES.get(k, k.replace("_", " ").title()) for k, _ in items
    ]
    values = [float(v) for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#4CAF50"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Score contribution",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_recommendation_card(card: dict, cluster: str = "") -> None:
    zone_type = str(card.get("zone_type", ""))
    zone_label = str(card.get("zone_name", card.get("zone_label", "")))
    score_progress = float(
        card.get("opportunity_score", card.get("score_progress", 0.0)) or 0.0
    )
    survival_risk = float(card.get("survival_risk", 0.0) or 0.0)
    confidence_bucket = str(card.get("confidence_bucket", ""))
    healthy_gap_summary = str(card.get("healthy_gap_summary", "") or "")
    freshness_note = str(card.get("freshness_note", "") or "")
    feature_contributions = card.get("feature_contributions") or {}
    recommended_subtype = str(card.get("recommended_subtype", "") or "")
    similar_restaurants = card.get("similar_restaurants") or []
    risk_flags = card.get("risk_flags") or []
    positive_drivers = card.get("positive_drivers") or []

    with st.container(border=True):
        col_badge, col_cluster = st.columns([3, 2])
        with col_badge:
            st.caption(_ZONE_TYPE_BADGE.get(zone_type, zone_type))
        with col_cluster:
            if cluster and cluster in _CLUSTER_BADGE:
                st.caption(_CLUSTER_BADGE[cluster])

        st.subheader(zone_label)
        if recommended_subtype:
            st.caption(f"Best fit: {recommended_subtype.replace('_', ' ').title()}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Opportunity Score", f"{score_progress * 100:.0f}%")
        col2.metric("Survival Risk", f"{survival_risk * 100:.0f}%")
        col3.metric(
            "Confidence",
            _CONFIDENCE_BADGE.get(confidence_bucket, confidence_bucket or "—"),
        )

        st.progress(max(0.0, min(1.0, score_progress)))

        summary = healthy_gap_summary.strip() or _build_gap_summary(card, cluster)
        st.write(summary)

        # Positive drivers
        if positive_drivers:
            with st.expander("Positive signals", expanded=False):
                for driver in positive_drivers[:4]:
                    st.success(driver, icon="✅")

        # Risk flags
        if risk_flags:
            with st.expander("Risk flags", expanded=False):
                for flag in risk_flags[:4]:
                    st.warning(flag, icon="⚠️")

        # Similar existing restaurants
        if similar_restaurants:
            st.caption(
                "Nearby comps: " + ", ".join(str(r) for r in similar_restaurants[:5])
            )

        with st.expander("Score breakdown"):
            _render_driver_chart(feature_contributions)

        if freshness_note:
            st.caption(freshness_note)
