"""Methodology page — interactive explanation of the three-phase pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_OUT = _REPO_ROOT / "data" / "output"


@st.cache_data(show_spinner=False)
def _load_phase1() -> pd.DataFrame:
    p = _DATA_OUT / "phase1_cluster_assignments.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_final() -> pd.DataFrame:
    p = _DATA_OUT / "final_recommendations.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_elbow() -> pd.DataFrame:
    p = _DATA_OUT / "phase1_elbow_table.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _render_cluster_scatter(df: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except ImportError:
        return
    if df.empty or "demand_score" not in df.columns:
        return

    COLORS = {
        "High Opportunity": "#e63946",
        "Established Hub": "#457b9d",
        "Growing Market": "#2a9d8f",
        "Low Demand": "#adb5bd",
    }
    fig = px.scatter(
        df,
        x="demand_score",
        y="gap_score",
        color="market_type",
        color_discrete_map=COLORS,
        size="halal_supply_rate",
        size_max=18,
        hover_data={"nta_id": True, "halal_supply_rate": ":.3f"},
        labels={
            "demand_score": "Demand Score",
            "gap_score": "Gap Score",
            "market_type": "Market Type",
        },
        title="Phase 1 — KMeans clusters (demand vs gap, bubble = supply rate)",
        height=380,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Market Type")
    st.plotly_chart(fig, use_container_width=True)


def _render_elbow_chart(df: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    if df.empty or "k" not in df.columns:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["k"], y=df["inertia"], mode="lines+markers", name="Inertia", yaxis="y1"
        )
    )
    if "silhouette" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["k"],
                y=df["silhouette"],
                mode="lines+markers",
                name="Silhouette",
                yaxis="y2",
                line=dict(dash="dot", color="orange"),
            )
        )
    fig.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 chosen")
    fig.update_layout(
        title="Elbow method — inertia + silhouette by k",
        xaxis=dict(title="k (clusters)", tickmode="linear"),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right", range=[0, 1]),
        height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_formula_sandbox(df: pd.DataFrame) -> None:
    """Interactive weight slider — adjust 0.4/0.4/0.2 and see top NTAs change."""
    if df.empty or not {"demand_score", "gap_score", "viability_score"}.issubset(
        df.columns
    ):
        return

    st.markdown("**Try different weights — see how the ranking changes:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_demand = st.slider(
            "Demand weight", 0.0, 1.0, 0.4, 0.05, key="w_demand_method"
        )
    with col2:
        w_gap = st.slider("Gap weight", 0.0, 1.0, 0.4, 0.05, key="w_gap_method")
    with col3:
        w_viability = st.slider(
            "Viability weight", 0.0, 1.0, 0.2, 0.05, key="w_viab_method"
        )

    total = w_demand + w_gap + w_viability
    if total == 0:
        st.warning("All weights are 0 — set at least one above 0.")
        return

    w_d, w_g, w_v = w_demand / total, w_gap / total, w_viability / total
    st.caption(
        f"Normalized: demand={w_d:.2f} · gap={w_g:.2f} · viability={w_v:.2f} (auto-scaled to sum 1.0)"
    )

    sandbox = df.copy()
    sandbox["custom_score"] = (
        w_d * sandbox["demand_score"]
        + w_g * sandbox["gap_score"]
        + w_v * sandbox["viability_score"]
    )
    sandbox["original_rank"] = (
        sandbox["final_score"].rank(ascending=False, method="min").astype(int)
    )
    sandbox["custom_rank"] = (
        sandbox["custom_score"].rank(ascending=False, method="min").astype(int)
    )
    sandbox["rank_delta"] = sandbox["original_rank"] - sandbox["custom_rank"]

    top5 = sandbox.nsmallest(5, "custom_rank")[
        [
            "nta_id",
            "market_type",
            "custom_score",
            "original_rank",
            "custom_rank",
            "rank_delta",
        ]
    ]
    top5.columns = [
        "NTA",
        "Market Type",
        "Custom Score",
        "Original Rank",
        "New Rank",
        "Δ Rank",
    ]
    top5["Δ Rank"] = top5["Δ Rank"].apply(
        lambda x: f"▲{x}" if x > 0 else (f"▼{abs(x)}" if x < 0 else "—")
    )
    st.dataframe(top5, use_container_width=True, hide_index=True)


def _render_risk_histogram(df: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except ImportError:
        return
    if df.empty or "high_risk_prob" not in df.columns:
        return

    fig = px.histogram(
        df,
        x="high_risk_prob",
        nbins=25,
        color="risk_bucket",
        color_discrete_map={
            "Low": "#2a9d8f",
            "Medium": "#f4a261",
            "High": "#e63946",
            "Unknown": "#adb5bd",
        },
        labels={
            "high_risk_prob": "GMM High-Risk Probability",
            "risk_bucket": "Risk Bucket",
        },
        title="Phase 3 — GMM risk probability distribution across all NTAs",
        height=280,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


def render_methodology_page() -> None:
    st.header("How This Works")
    st.write(
        "Three phases of analysis, each building on the previous. "
        "The interactive demos below let you explore the actual data — not just descriptions."
    )

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Phase 1 — Halal Market Typing")
    st.write(
        "Each NYC NTA is clustered into one of four halal market types using **KMeans** "
        "(implemented from scratch in NumPy — no sklearn). "
        "Features are z-score normalized before clustering."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Features:**")
        st.markdown("- `demand_score` — Bayesian-shrunk halal review share (prior=10)")
        st.markdown("- `halal_supply_rate` — halal cuisine density proxy")
        st.markdown("- `gap_score` — max(demand − supply, 0)")
    with col2:
        st.markdown("**Market types:**")
        st.markdown("- 🔴 **High Opportunity** — high demand, low supply")
        st.markdown("- 🔵 **Established Hub** — strong existing halal scene")
        st.markdown("- 🟢 **Growing Market** — moderate demand, little supply")
        st.markdown("- ⚫ **Low Demand** — limited halal activity")

    phase1_df = _load_phase1()
    elbow_df = _load_elbow()

    if not phase1_df.empty:
        tab_scatter, tab_elbow = st.tabs(["Cluster Scatter", "Elbow / Silhouette"])
        with tab_scatter:
            _render_cluster_scatter(phase1_df)
        with tab_elbow:
            _render_elbow_chart(elbow_df)
            st.caption(
                "k=4 selected: silhouette score **0.3963**. Large inertia drop from k=2→3→4, smaller after."
            )
    else:
        st.caption("Run `scripts/run_phase1.py` to generate cluster data.")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Phase 2 — Opportunity Scoring & Interactive Sandbox")
    st.write("Each NTA gets a composite score combining three independent signals.")
    st.code(
        "final_score = 0.4 × demand_score + 0.4 × gap_score + 0.2 × viability_score",
        language="python",
    )
    st.markdown("**Viability** — rule-based from NYC inspection records:")
    st.markdown("- −0.5 if critical violation rate > 75th percentile")
    st.markdown("- −0.5 if Grade A rate < 25th percentile")
    st.caption("Spearman rank correlation (gap vs final score): 0.93")

    final_df = _load_final()
    if not final_df.empty:
        with st.expander(
            "🎛️ Formula Sandbox — adjust weights and see the top NTAs change",
            expanded=True,
        ):
            _render_formula_sandbox(final_df)
    else:
        st.caption("Run the full pipeline to enable the formula sandbox.")

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Phase 3 — Risk & Forecast Layer")
    st.write(
        "Phase 3 adds supplementary signals. "
        "It does **not** change the Phase 2 `final_score` ranking — it adds "
        "`high_risk_prob` and `halal_demand_forecast` as context layers."
    )
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**GMM Risk Clustering** (n=4 components)")
        st.markdown(
            "- 5 features: critical_rate, grade_a_rate, inspection_freq, demand, supply"
        )
        st.markdown("- `high_risk_prob` = P(top-2 risk components)")
        st.markdown("- BIC supports n=4 · Silhouette: 0.40")
    with col4:
        st.markdown("**Ridge Demand Forecast**")
        st.markdown("- 2022 features → predict 2023 halal review share")
        st.markdown(
            "- R² = 0.16 (directional only — barely beats persistence baseline)"
        )
        st.markdown("- 5-fold CV + ablation table to verify feature contributions")

    if not final_df.empty:
        _render_risk_histogram(final_df)
    else:
        st.caption("Run `scripts/run_phase3.py` to generate risk data.")

    # ── Data sources ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Data Sources")
    st.markdown(
        "| Source | Purpose |\n"
        "|---|---|\n"
        "| Yelp reviews + Gemini labels | Halal demand signal (14,853 reviews, 144 NTAs) |\n"
        "| CAMIS / NYC DOHMH records | Halal supply signal (494 halal restaurants) |\n"
        "| NYC DOHMH inspection records | Viability and GMM risk signals |\n"
        "| US Census / NTA boundaries | Geographic unit definition |"
    )

    # ── Caveats ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Caveats")
    st.warning(
        "All signals are proxies, not ground truth. Interpret with caution.", icon="⚠️"
    )
    st.markdown(
        "1. **`demand_score`** is a Yelp discussion proxy — Bayesian shrinkage stabilizes small-sample NTAs "
        "(prior=10 reviews). Not a true consumer demand measurement.\n"
        "2. **`halal_supply_rate`** uses cuisine family as a proxy (Pakistani, Middle Eastern, etc.) — "
        "not a certified halal count. Likely undercounts by ~20–40%.\n"
        "3. **`gap_score`** clips negative values to 0 — oversupplied areas all score 0, losing relative signal.\n"
        "4. **Risk scores** reflect the general restaurant environment, not halal-specific risk.\n"
        "5. **Ridge R²=0.16** is a dataset-size ceiling, not a model failure — only ~50 NTAs have "
        "sufficient review data in both 2022 and 2023 for the forecast join."
    )


render_methodology_page()
