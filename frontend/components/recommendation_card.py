"""Recommendation-card UI helpers."""

from __future__ import annotations

import streamlit as st


_CONFIDENCE_BADGE: dict[str, str] = {
    "high": "🟢 High",
    "medium": "🟡 Medium",
    "low": "🔴 Low",
}


def _score_progress(score: float) -> float:
    """Normalize an opportunity score to [0.0, 1.0] for st.progress."""
    if score <= 1.0:
        return float(score)
    if score <= 10.0:
        return score / 10.0
    if score <= 100.0:
        return score / 100.0
    return 1.0


def render_recommendation_card(card: dict[str, object]) -> None:
    """Render a single recommendation card with full detail."""

    with st.container(border=True):
        col_title, col_badge = st.columns([3, 1])
        with col_title:
            st.markdown(f"### {card.get('zone_name', 'Unknown Zone')}")
        with col_badge:
            confidence = str(card.get("confidence_bucket", "low"))
            badge_text = _CONFIDENCE_BADGE.get(confidence, f"❓ {confidence}")
            st.markdown(f"**{badge_text}**")

        score = float(card.get("opportunity_score", 0.0))
        st.markdown(f"**Opportunity Score:** {score:.2f}")
        st.progress(_score_progress(score))

        gap_summary = str(card.get("healthy_gap_summary", ""))
        if gap_summary:
            st.info(gap_summary)

        positives: list[str] = list(card.get("positives", []))  # type: ignore[arg-type]
        risks: list[str] = list(card.get("risks", []))  # type: ignore[arg-type]

        col_pos, col_risk = st.columns(2)
        with col_pos:
            if positives:
                st.markdown("**Positives**")
                for item in positives:
                    st.markdown(f"✓ {item}")
        with col_risk:
            if risks:
                st.markdown("**Risks**")
                for item in risks:
                    st.markdown(f"⚠ {item}")

        freshness = str(card.get("freshness_note", ""))
        if freshness:
            st.caption(freshness)
