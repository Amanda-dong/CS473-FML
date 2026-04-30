from __future__ import annotations

import csv
import io

import streamlit as st

from frontend.components.recommendation_card import render_recommendation_card

_CSV_COLUMNS = [
    "zone_name",
    "zone_type",
    "opportunity_score",
    "confidence_bucket",
    "survival_risk",
    "healthy_gap_summary",
]


def _make_csv(recommendations: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(recommendations)
    return buf.getvalue().encode("utf-8")


def _render_summary_row(recommendations: list[dict]) -> None:
    """Show aggregate stats across the returned shortlist."""
    scores = [float(r.get("opportunity_score", 0.0) or 0.0) for r in recommendations]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    conf_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in recommendations:
        bucket = str(r.get("confidence_bucket", "")).lower()
        if bucket in conf_counts:
            conf_counts[bucket] += 1

    conf_label = f"{conf_counts['high']} high / {conf_counts['medium']} medium / {conf_counts['low']} low"

    c1, c2, c3 = st.columns(3)
    c1.metric("Zones shown", len(recommendations))
    c2.metric("Avg. opportunity score", f"{avg_score * 100:.0f}%")
    c3.metric("Confidence mix", conf_label)


def render_top_match_panel(
    user_state: dict,
    recommendation: dict | None,
    cluster_map: dict[str, str] | None = None,
) -> str | None:
    """Render a featured top match panel and return its zone_id."""
    if not recommendation:
        return None

    zone_id = str(recommendation.get("zone_id", ""))
    zone_name = str(
        recommendation.get("zone_name", recommendation.get("zone_label", "Top Match"))
    )
    score = float(recommendation.get("opportunity_score", 0.0) or 0.0)
    survival_risk = float(recommendation.get("survival_risk", 0.0) or 0.0)
    confidence = str(recommendation.get("confidence_bucket", "—")).title()
    concept = str(user_state.get("concept_subtype", "")).replace("_", " ").title()
    zone_type = str(recommendation.get("zone_type", ""))
    cluster = (cluster_map or {}).get(zone_type, "")
    summary = str(recommendation.get("healthy_gap_summary", "") or "")
    positives = (
        recommendation.get("positive_drivers") or recommendation.get("positives") or []
    )

    st.subheader("Top Match")
    st.caption("Start here. This is the strongest match for your current query.")
    with st.container(border=True):
        st.markdown(f"#### {zone_name}")
        meta_bits = [
            bit
            for bit in [
                zone_type.replace("_", " ").title() if zone_type else "",
                cluster.title() if cluster else "",
            ]
            if bit
        ]
        if meta_bits:
            st.caption(" | ".join(meta_bits))

        m1, m2, m3 = st.columns(3)
        m1.metric("Opportunity score", f"{score * 100:.0f}%")
        m2.metric("Survival risk", f"{survival_risk * 100:.0f}%")
        m3.metric("Confidence", confidence)

        if concept:
            st.markdown(
                f"**Why it stands out:** {summary or f'This zone ranked first for {concept}.'}"
            )
        if positives:
            st.markdown(
                "**Key strengths:** " + "; ".join(str(item) for item in positives[:3])
            )
    return zone_id or None


def render_results_panel(
    user_state: dict,
    recommendations: list[dict] | None = None,
    cluster_map: dict[str, str] | None = None,
    featured_zone_id: str | None = None,
) -> None:
    if recommendations is None:
        st.info("Configure your search in the sidebar to see recommendations.")
        return

    if recommendations == []:
        st.warning("No recommendations found for the current filters.")
        return

    st.subheader("Recommended Zones")
    st.caption("Use these cards to compare the rest of your shortlist.")

    _render_summary_row(recommendations)
    st.divider()

    remaining = [
        rec
        for rec in recommendations
        if str(rec.get("zone_id", "")) != str(featured_zone_id or "")
    ]
    if featured_zone_id and remaining:
        st.caption("Compare the remaining options below.")
    elif featured_zone_id and not remaining:
        st.caption("Your query returned one highlighted recommendation.")

    for rec in remaining if featured_zone_id else recommendations:
        zone_type = rec.get("zone_type", "")
        cluster = (cluster_map or {}).get(zone_type, "")
        render_recommendation_card(rec, cluster=cluster)

    csv_bytes = _make_csv(recommendations)
    st.download_button(
        "📥 Export shortlist as CSV",
        data=csv_bytes,
        file_name="shortlist.csv",
        mime="text/csv",
    )
