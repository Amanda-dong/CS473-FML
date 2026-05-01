"""Recommendation card — displays one neighborhood recommendation card."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from frontend.review_evidence import (
    clip_review,
    evidence_csv_path,
    nta_review_counts,
    sample_reviews_for_nta,
)

MARKET_TYPE_EMOJI = {
    "High Opportunity": "🔴",
    "Established Hub": "🔵",
    "Growing Market": "🟢",
    "Low Demand": "⚫",
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REVIEWS_PATH = _REPO_ROOT / "data" / "raw" / "yelp_reviews_with_zones.csv"
_ZONE_LABELS = {
    "bk-tandon": "NYU Tandon / MetroTech",
    "bk-downtownbk": "Downtown Brooklyn",
    "bk-williamsburg": "Williamsburg",
    "bk-navy-yard": "Brooklyn Navy Yard / Vinegar Hill",
    "bk-fort-greene": "Fort Greene / Pratt Area",
    "bk-crown-hts": "Crown Heights",
    "bk-sunset-pk": "Sunset Park",
    "mn-midtown-e": "Midtown East",
    "mn-fidi": "Financial District",
    "mn-columbia": "Morningside Heights / Columbia",
    "mn-nyu-wash-sq": "Washington Square / NYU",
    "mn-ues-hosp": "Upper East Side / Hospital Row",
    "mn-chelsea": "Chelsea / Hudson Yards",
    "mn-harlem": "Harlem",
    "mn-lic-adj": "East Midtown / UN",
    "qn-lic": "Long Island City",
    "qn-astoria": "Astoria",
    "qn-flushing": "Flushing",
    "qn-jackson-hts": "Jackson Heights",
    "qn-forest-hills": "Forest Hills",
    "qn-jamaica": "Jamaica",
    "bx-fordham": "Fordham",
    "bx-mott-haven": "Mott Haven",
    "bx-co-op-city": "Co-op City",
    "bx-tremont": "East Tremont",
    "si-st-george": "St. George",
    "si-new-spring": "New Springville",
}


@st.cache_data(show_spinner=False)
def _load_nta_zone_lookup() -> dict[str, str]:
    if not _REVIEWS_PATH.exists():
        return {}
    try:
        df = pd.read_csv(_REVIEWS_PATH, usecols=["nta", "zone_id"])
    except Exception:
        return {}

    df = df.dropna(subset=["nta", "zone_id"]).copy()
    if df.empty:
        return {}

    df["nta"] = df["nta"].astype(str).str.strip()
    df["zone_id"] = df["zone_id"].astype(str).str.strip()
    df = df[(df["nta"] != "") & (df["zone_id"] != "")]
    if df.empty:
        return {}

    top_zone = (
        df.groupby(["nta", "zone_id"])
        .size()
        .reset_index(name="n")
        .sort_values(["nta", "n"], ascending=[True, False])
        .drop_duplicates(subset=["nta"])
    )
    return dict(zip(top_zone["nta"], top_zone["zone_id"]))


def _borough(nta_id: str) -> str:
    return BOROUGH_NAME.get(str(nta_id)[:2].upper(), "NYC")


def _prettify_zone_id(zone_id: str) -> str:
    zone_key = str(zone_id).strip().lower()
    if not zone_key:
        return ""
    if zone_key in _ZONE_LABELS:
        return _ZONE_LABELS[zone_key]
    if zone_key.startswith("nta-"):
        return ""
    return zone_key.replace("-", " ").title()


def _display_name(nta_id: str) -> str:
    zone_lookup = _load_nta_zone_lookup()
    zone_id = zone_lookup.get(str(nta_id).strip(), "")
    label = _prettify_zone_id(zone_id)
    if label:
        return label
    code = str(nta_id).strip()
    return f"{_borough(code)} ({code})" if code else "NYC"


def _format_similar_neighborhoods(similar_ntas: list[str]) -> str:
    return " · ".join(_display_name(nta) for nta in similar_ntas)


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
            st.markdown(f"### {_display_name(nta_id)}")
            st.caption(f"{_borough(nta_id)} · {nta_id}")
        with col_badge:
            st.markdown(f"**{emoji} {market_type}**")

        # Main scores
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Overall fit",
            _fmt_score(final_score),
            help="This is the main overall ranking score for this neighborhood.",
        )
        c2.metric(
            "Local interest",
            _signal_label(demand_score),
            help="A review-based signal for halal interest in the area.",
        )
        c3.metric(
            "Open space",
            _signal_label(gap_score),
            help="A simple read on how under-served the area may be for halal food.",
        )

        st.progress(float(final_score) if final_score else 0.0)
        st.caption(
            "Quick read: a stronger overall fit means a better rank. Risk is shown separately below."
        )

        # Yelp / Gemini labeled review evidence for this zone
        with st.expander("Review evidence — sample Yelp rows (Gemini labels)", expanded=False):
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
                st.markdown(
                    f"| Pipeline metric | Value |\n|--|--|\n"
                    f"| **`demand_score`** (Opportunity ranking + clustering) | **{_fmt_score(demand_score)}** |\n"
                    f"| **Market type** (k-means on demand/supply/gap · not snippet counts) | **{market_type}** |"
                )
                st.info(
                    "**Counts below** only reflect rows stored in **`gemini_labels_full.csv`** for "
                    f"this NTA. **`demand_score`** is computed in **`halal_demand.py`** from the **full Yelp join** "
                    "per neighborhood, **shrunk-share → min–max across all NTAs**. "
                    "This is why the sample review counts below do not exactly match the ranking metrics above."
                )

                counts = nta_review_counts(review_pool, nta_id)
                if counts["total"] == 0:
                    st.caption(f"No review rows mapped to **`{nta_id}`** in the labeled Yelp export.")
                else:
                    _other_note = ""
                    if counts.get("other_labels", 0) > 0:
                        _other_note = (
                            f" · **{counts['other_labels']}** rows tagged with another "
                            "**`halal_relevance`** value"
                        )
                    st.caption(
                        f"In the labeled Yelp sample for **`{nta_id}`**: "
                        f"**{counts['explicit_halal']}** explicit-halal · "
                        f"**{counts['implicit_halal']}** implicit-halal · "
                        f"**{counts['not_related']}** not halal-related (`not_related`). "
                        f"**{counts['total']}** review rows · **{counts['unique_venues']}** distinct venues"
                        f"{_other_note}."
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
            st.caption("No halal-relevant restaurants are currently recorded here.")
        else:
            st.caption(
                f"Current halal presence: {diversity} cuisine type{'s' if diversity > 1 else ''} recorded nearby."
            )

        # Risk section
        with st.expander("Risk", expanded=False):
            try:
                risk_prob = float(high_risk_prob)
            except (TypeError, ValueError):
                risk_prob = 0.5

            risk_icon = RISK_ICONS.get(risk_bucket, "❓")

            st.metric(
                "Neighborhood risk",
                f"{risk_icon} {risk_bucket}",
                help="This reflects the local operating environment, not your specific business plan.",
            )
            st.progress(risk_prob)
            st.caption(
                f"Estimated chance of a tougher operating environment: {risk_prob:.0%}."
            )

            if risk_confidence == "Low confidence":
                st.warning(
                    "Fewer than 10 inspection records were available here, so treat this risk reading cautiously.",
                    icon="⚠️",
                )

        # Phase 3 insight
        with st.expander("Next-Year Outlook", expanded=False):
            if halal_demand_forecast is not None:
                try:
                    val = float(halal_demand_forecast)
                    if val > 0.5:
                        st.success(
                            f"Interest appears to be trending up next year. Projected halal discussion share: {val:.1%}"
                        )
                    elif val > 0.3:
                        st.info(
                            f"Interest looks fairly steady next year. Projected halal discussion share: {val:.1%}"
                        )
                    else:
                        st.warning(
                            f"Interest looks weaker next year. Projected halal discussion share: {val:.1%}"
                        )
                except (TypeError, ValueError):
                    st.caption("Forecast not available for this neighborhood.")
            else:
                st.caption("Forecast not available for this neighborhood.")
            st.caption("Use this as a rough directional signal, not a precise prediction.")

        # Similar NTAs
        if similar_ntas:
            st.caption("Similar areas: " + _format_similar_neighborhoods(similar_ntas))
