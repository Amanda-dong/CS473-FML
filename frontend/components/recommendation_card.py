"""Recommendation card — displays one NTA as a structured card."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from frontend.review_evidence import clip_review, evidence_csv_path, nta_review_counts, sample_reviews_for_nta

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

RISK_ICONS = {
    "Low": "✅",
    "Medium": "⚠️",
    "High": "🔴",
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


def _signal_label(val) -> str:
    try:
        v = float(val)
        if v >= 0.7:
            return "Very Strong"
        if v >= 0.4:
            return "Moderate"
        if v >= 0.2:
            return "Weak"
        return "Low"
    except (TypeError, ValueError):
        return "—"


def _halal_rel_badge(label: str) -> str:
    label = str(label).strip().lower()
    return {
        "explicit_halal": "Explicit halal mention",
        "implicit_halal": "Implicit halal context",
        "not_related": "Not halal-related",
    }.get(label, label.replace("_", " ").title())


def render_recommendation_card(
    row: dict,
    rank: int,
    *,
    review_pool: pd.DataFrame | None = None,
    repo_root: Path | None = None,
) -> None:
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
    critical_rate = row.get("critical_rate", 0.0)
    grade_a_rate = row.get("grade_a_rate", 0.0)
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
            _signal_label(demand_score),
            help="Share of Yelp reviews mentioning halal (proxy, not true demand)",
        )
        c3.metric(
            "Opportunity Gap",
            _signal_label(gap_score),
            help="Demand proxy minus supply proxy (heuristic estimate)",
        )

        st.progress(float(final_score) if final_score else 0.0)

        # Yelp / Gemini labeled review evidence for this zone
        with st.expander("Review evidence — what diners wrote (Yelp × Gemini)", expanded=False):
            if review_pool is None or review_pool.empty:
                hint = ""
                if repo_root is not None:
                    hint = (
                        f"Expected file missing: `{evidence_csv_path(repo_root)}`. "
                        "Place Gemini-labeled review export under `data/raw/`."
                    )
                st.info(
                    "No review evidence loaded. Run the dashboard from the repo with "
                    "`data/raw/gemini_labels_full.csv` present, or check the path.\n\n" + hint
                )
            else:
                counts = nta_review_counts(review_pool, nta_id)
                if counts["total"] == 0:
                    st.caption(f"No review rows mapped to **`{nta_id}`** in the labeled Yelp export.")
                else:
                    st.caption(
                        f"In the labeled Yelp sample for **`{nta_id}`**: "
                        f"**{counts['explicit_halal']}** explicit-halal, "
                        f"**{counts['implicit_halal']}** implicit-halal, "
                        f"**{counts['total']}** review rows · "
                        f"**{counts['unique_venues']}** distinct venues."
                    )
                    samples = sample_reviews_for_nta(review_pool, nta_id, k=6)
                    if samples.empty:
                        st.caption("No rows to preview after sorting.")
                    else:
                        st.caption(
                            "Up to **6 unique venues** — one prioritized review each "
                            "(explicit / implicit halal first, then by rating)."
                        )

                        display_rows = []
                        for _, rr in samples.iterrows():
                            name = rr.get("business_name") or rr.get("restaurant_id") or "Unknown venue"
                            name = str(name).strip() or "Unknown venue"
                            rt = rr.get("rating")
                            rt_txt = f"★ {float(rt):.0f}" if pd.notna(rt) else ""
                            rel = rr.get("halal_relevance", "")
                            txt = clip_review(str(rr.get("review_text", "")), max_chars=400)
                            display_rows.append(
                                {
                                    "Venue": name[:80],
                                    "Rating": rt_txt,
                                    "Halal label": _halal_rel_badge(rel),
                                    "Review excerpt": txt,
                                }
                            )
                        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        # Supply info
        diversity = int(halal_cuisine_diversity) if halal_cuisine_diversity else 0
        if diversity == 0:
            st.caption("No halal-relevant restaurants currently recorded in this area.")
        else:
            st.caption(
                f"Halal-relevant restaurants present · {diversity} cuisine type{'s' if diversity > 1 else ''}"
            )

        # Risk section
        with st.expander("Risk & Environment", expanded=False):
            try:
                float(high_risk_prob)
            except (TypeError, ValueError):
                pass

            risk_icon = RISK_ICONS.get(risk_bucket, "❓")
            risk_description = {
                "Low": "This neighborhood has a relatively clean inspection record. Lower regulatory risk for new restaurant operators.",
                "Medium": "This neighborhood has moderate inspection risk. Some history of violations — worth reviewing before committing.",
                "High": "This neighborhood has a high rate of critical inspection violations. Factor this into your operational planning.",
                "Unknown": "Insufficient inspection data available for this neighborhood.",
            }.get(risk_bucket, "")

            st.markdown(f"### {risk_icon} {risk_bucket} Risk")
            st.write(risk_description)

            if risk_confidence == "Low confidence":
                st.caption(
                    "⚠️ Based on fewer than 10 inspection records — treat with caution."
                )
            else:
                st.caption("Based on NYC DOHMH restaurant inspection records.")

        # Phase 3 insight
        with st.expander("Next-Year Demand Outlook", expanded=False):
            if halal_demand_forecast is not None:
                try:
                    val = float(halal_demand_forecast)
                    if val > 0.5:
                        st.success(
                            f"Demand trending up — projected halal discussion share: {val:.1%}"
                        )
                    elif val > 0.3:
                        st.info(
                            f"Demand stable — projected halal discussion share: {val:.1%}"
                        )
                    else:
                        st.warning(
                            f"Demand signal weak — projected halal discussion share: {val:.1%}"
                        )
                except (TypeError, ValueError):
                    st.caption("Forecast not available for this NTA.")
            else:
                st.caption("Forecast not available for this NTA.")
            st.caption("R² = 0.16 — treat as directional signal only, not a precise forecast.")

        # Similar NTAs
        if similar_ntas:
            st.caption("Similar neighborhoods: " + " · ".join(similar_ntas))
