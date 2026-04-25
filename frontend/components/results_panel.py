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


def render_results_panel(
    user_state: dict,
    recommendations: list[dict] | None = None,
    cluster_map: dict[str, str] | None = None,
) -> None:
    if recommendations is None:
        st.info("Configure your search in the sidebar to see recommendations.")
        return

    if recommendations == []:
        st.warning("No recommendations found for the current filters.")
        return

    concept = user_state.get("concept_subtype", "")
    if concept:
        st.caption(f"Concept: {concept.replace('_', ' ').title()}")

    _render_summary_row(recommendations)
    st.divider()

    for rec in recommendations:
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
