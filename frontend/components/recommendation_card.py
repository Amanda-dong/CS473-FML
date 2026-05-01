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
    "fast-growing": "📈 Fast-Growing",
    "gentrifying": "📈 Fast-Growing",
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


def _render_driver_chart(feature_contributions: dict, chart_key: str) -> None:
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
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _render_evidence_snapshot(
    feature_contributions: dict, positive_drivers: list, risk_flags: list
) -> None:
    """Render a compact evidence line with minimal UI blocks."""
    drivers_text = "Unavailable"
    if feature_contributions:
        ordered = sorted(
            feature_contributions.items(),
            key=lambda kv: abs(float(kv[1])),
            reverse=True,
        )
        top = ordered[:2]
        if top:
            labels = [
                FEATURE_DISPLAY_NAMES.get(k, k.replace("_", " ").title()) for k, _ in top
            ]
            drivers_text = ", ".join(labels)

    positive_text = positive_drivers[0] if positive_drivers else "Not available"
    risk_text = risk_flags[0] if risk_flags else "Not available"

    st.markdown("**Evidence**")
    st.caption(f"Model drivers: {drivers_text}")
    st.caption(f"Positive signal: {positive_text}")
    st.caption(f"Risk signal: {risk_text}")


def render_recommendation_card(
    card: dict, cluster: str = "", key_namespace: str = ""
) -> None:
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
    risk_flags = card.get("risk_flags") or card.get("risks") or []
    positive_drivers = card.get("positive_drivers") or card.get("positives") or []
    scoring_path = str(card.get("scoring_path", "unknown") or "unknown")
    model_version = str(card.get("model_version", "unknown") or "unknown")

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
        st.caption(f"Scoring path: {scoring_path} | Model: {model_version}")

        st.progress(max(0.0, min(1.0, score_progress)))
        st.caption(
            "Interpretation: higher opportunity suggests stronger fit; lower survival risk suggests safer execution."
        )

        summary = healthy_gap_summary.strip() or _build_gap_summary(card, cluster)
        st.markdown(f"**Model summary:** {summary}")
        _render_evidence_snapshot(feature_contributions, positive_drivers, risk_flags)

        # Similar existing restaurants
        if similar_restaurants:
            st.caption(
                "Nearby comps: " + ", ".join(str(r) for r in similar_restaurants[:5])
            )

        if freshness_note:
            st.caption(f"Data source: {freshness_note}")
