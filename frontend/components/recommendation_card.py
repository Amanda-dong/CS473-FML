"""Recommendation card — displays one neighborhood recommendation card."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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


def _build_radar_chart(
    demand: float,
    gap: float,
    viability: float,
    risk_prob: float,
    forecast_norm: float,
) -> "go.Figure":
    """Radar chart: 5 opportunity dimensions, all on [0, 1]."""
    import plotly.graph_objects as go

    categories = [
        "Halal Demand",
        "Market Gap",
        "Operating Safety",
        "Low Risk",
        "Future Trend",
    ]
    values = [
        float(demand or 0),
        float(gap or 0),
        float(viability or 0),
        1.0 - float(risk_prob or 0.5),
        float(forecast_norm or 0.5),
    ]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor="rgba(255,75,75,0.15)",
            line=dict(color="rgba(255,75,75,0.8)", width=2),
            name="Profile",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1], tickfont=dict(size=9), showticklabels=False
            ),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _opportunity_summary(
    demand: float,
    gap: float,
    viability: float,
    risk_bucket: str,
    market_type: str,
    diversity: int,
) -> str:
    """Generate plain-English 2-sentence opportunity summary from signal values."""
    d = float(demand or 0)
    g = float(gap or 0)
    v = float(viability or 0)

    demand_txt = (
        "strong halal interest"
        if d >= 0.5
        else ("moderate halal interest" if d >= 0.25 else "low halal awareness")
    )
    gap_txt = (
        "few existing halal options"
        if g >= 0.5
        else ("some competition" if g >= 0.2 else "an established halal scene")
    )
    risk_txt = {
        "Low": "low operating risk",
        "Medium": "moderate inspection risk",
        "High": "elevated regulatory risk",
    }.get(risk_bucket, "unknown risk level")
    diversity_txt = (
        f"{diversity} halal cuisine type{'s' if diversity != 1 else ''} already present"
        if diversity > 0
        else "no recorded halal presence"
    )

    sentence1 = f"This area shows **{demand_txt}** with **{gap_txt}**, placing it in the **{market_type}** segment."
    sentence2 = f"Operating environment has **{risk_txt}** — {diversity_txt}."
    return sentence1 + "  \n" + sentence2


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
    halal_demand_forecast_norm = row.get("halal_demand_forecast_norm", None)
    similar_ntas_raw = str(row.get("similar_ntas", "") or "")
    similar_ntas = [s.strip() for s in similar_ntas_raw.split(",") if s.strip()]
    latent_demand_score = row.get('latent_demand_score', None)
    cluster_confidence = row.get('cluster_confidence', None)

    badge_class = market_type.lower().replace(" ", "-")

    with st.container():
        # Header
        col_title, col_badge = st.columns([5, 2])
        with col_title:
            st.markdown(f"### #{rank} {_display_name(nta_id)}")
            st.caption(f"{_borough(nta_id)} · {nta_id}")
        with col_badge:
            _mc = {
                'High Opportunity': ('#fde8e8', '#e63946', '#c0392b'),
                'Established Hub':  ('#e3edf7', '#457b9d', '#2c5f7a'),
                'Growing Market':   ('#e8f5e9', '#2a9d8f', '#1e7a6e'),
                'Low Demand':       ('#f5f5f5', '#adb5bd', '#6c757d'),
            }
            mc_bg, mc_border, mc_text = _mc.get(market_type, ('#f5f5f5', '#adb5bd', '#6c757d'))
            emoji = MARKET_TYPE_EMOJI.get(market_type, "⚪")
            st.markdown(
                f'<span style="background:{mc_bg};border:1.5px solid {mc_border};border-radius:20px;'
                f'padding:4px 12px;font-size:0.82em;font-weight:600;color:{mc_text};display:inline-block;">'
                f'{emoji} {market_type}</span>',
                unsafe_allow_html=True,
            )

        # Radar + Score Gauge
        col_radar, col_gauge = st.columns([2, 2])
        with col_radar:
            try:
                fig = _build_radar_chart(
                    demand_score,
                    gap_score,
                    viability_score,
                    high_risk_prob,
                    halal_demand_forecast_norm,
                )
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}, key=f"radar_{nta_id}"
                )
            except Exception:
                pass
        
        with col_gauge:
            pass # Gauge moved to consolidated block below

        # Consolidated Metrics & Gauge
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            'Overall fit',
            _fmt_score(final_score),
            help='Main ranking score: 0.4×demand + 0.4×gap + 0.2×viability.',
        )
        c2.metric(
            'Demand',
            _signal_label(demand_score),
            help='Bayesian-shrunk halal review share across Yelp data.',
        )
        c3.metric(
            'Gap',
            _signal_label(gap_score),
            help='max(demand − supply, 0) — unmet demand proxy.',
        )
        c4.metric(
            'Latent Demand',
            _signal_label(latent_demand_score) if latent_demand_score is not None else '—',
            help='Implicit halal interest + keyword signals. Captures demand where halal restaurants are absent.',
        )
        if cluster_confidence is not None:
            try:
                cc = float(cluster_confidence)
                if cc < 0.25:
                    st.warning(f'⚠ Borderline cluster (confidence {cc:.2f}) — market type may shift with new data.')
            except (TypeError, ValueError):
                pass
        try:
            import plotly.graph_objects as _go
            _score_val = float(final_score) * 100 if final_score else 0.0
            _gauge = _go.Figure(_go.Indicator(
                mode='gauge+number',
                value=_score_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'size': 9}},
                    'bar': {'color': '#1a472a'},
                    'steps': [
                        {'range': [0, 40], 'color': '#fde8e8'},
                        {'range': [40, 70], 'color': '#fff8e1'},
                        {'range': [70, 100], 'color': '#e8f5e9'},
                    ],
                    'threshold': {'line': {'color': '#e9c46a', 'width': 3}, 'thickness': 0.8, 'value': _score_val},
                },
                number={'suffix': '/100', 'font': {'size': 14}},
                title={'text': 'Overall Score', 'font': {'size': 11}},
            ))
            _gauge.update_layout(height=130, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(_gauge, use_container_width=True, config={'displayModeBar': False})
        except Exception:
            st.progress(float(final_score) if final_score else 0.0)
        st.caption('Score = 0.4×demand + 0.4×gap + 0.2×viability. Radar shows 5 dimensions.')

        # Plain-English summary
        st.info(_opportunity_summary(
            demand_score,
            gap_score,
            viability_score,
            risk_bucket,
            market_type,
            int(halal_cuisine_diversity or 0),
        ))

        with st.expander('Score breakdown', expanded=False):
            try:
                contrib_fig = go.Figure(go.Bar(
                    x=[0.4 * float(demand_score), 0.4 * float(gap_score), 0.2 * float(viability_score)],
                    y=['Demand signal', 'Supply gap', 'Viability'],
                    orientation='h',
                    marker_color=['#e63946', '#2a9d8f', '#457b9d'],
                    text=[f'{0.4*float(demand_score):.3f}', f'{0.4*float(gap_score):.3f}', f'{0.2*float(viability_score):.3f}'],
                    textposition='outside',
                ))
                contrib_fig.update_layout(
                    xaxis=dict(range=[0, 0.42], title='Contribution to score'),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=150,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#fafafa"}
                )
                st.plotly_chart(contrib_fig, use_container_width=True, config={'displayModeBar': False}, key=f"breakdown_{nta_id}")
            except Exception as exc:
                st.info(f"Score breakdown unavailable for this row ({exc}).")

        # Yelp / Gemini labeled review evidence for this zone
        with st.expander(
            "Review evidence — sample Yelp rows (Gemini labels)", expanded=False
        ):
            if review_pool is None or review_pool.empty:
                st.info("No review evidence loaded.")
            else:
                counts = nta_review_counts(review_pool, nta_id)
                if counts["total"] == 0:
                    st.caption(f"No review rows mapped to **{nta_id}**.")
                else:
                    samples = sample_reviews_for_nta(review_pool, nta_id, k=6)
                    display_rows = []
                    for _, rr in samples.iterrows():
                        name = rr.get("business_name") or rr.get("restaurant_id") or "Unknown venue"
                        rt = rr.get("rating")
                        rt_txt = f"★ {float(rt):.0f}" if pd.notna(rt) else ""
                        rel = rr.get("halal_relevance", "")
                        txt = clip_review(str(rr.get("review_text", "")), max_chars=400)
                        display_rows.append({
                            "Venue": str(name)[:80],
                            "Rating": rt_txt,
                            "Label": _halal_rel_badge(rel),
                            "Review": txt,
                        })
                    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        # Risk section
        with st.expander("Risk Detail", expanded=False):
            risk_icon = RISK_ICONS.get(risk_bucket, "❓")
            st.markdown(f"**Status**: {risk_icon} {risk_bucket}")
            st.progress(1.0 - float(viability_score))
            if risk_confidence == "Low confidence":
                st.warning("Fewer than 10 records available — treat with caution.")

        # Phase 3 insight
        with st.expander("Next-Year Outlook", expanded=False):
            if halal_demand_forecast is not None:
                val = float(halal_demand_forecast)
                if val > 0.5: st.success(f"Trending Up: {val:.1%}")
                elif val > 0.3: st.info(f"Steady: {val:.1%}")
                else: st.warning(f"Weak: {val:.1%}")
            else:
                st.caption("Forecast not available.")

        # Similar NTAs
        if similar_ntas:
            with st.expander("Similar neighborhoods", expanded=False):
                for nta in similar_ntas[:3]:
                    st.markdown(f"- **{_display_name(nta)}** ({nta})")
