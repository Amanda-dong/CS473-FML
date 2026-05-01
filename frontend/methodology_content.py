"""Methodology content for the main Streamlit app."""

from __future__ import annotations

import streamlit as st

from pathlib import Path
_img = Path(__file__).resolve().parent.parent / 'assets' / 'approach_diagram.png'
if _img.exists():
    st.image(str(_img))


def render_methodology_page() -> None:
    st.header("Introduction")

    st.markdown("### The Problem")
    st.markdown(
        """
        Independent operators face an information gap when choosing NYC locations.

        - Chains can commission site-selection analytics; independents often rely on intuition.
        - NYC has **195 NTAs**, but opportunity is usually hidden at neighborhood granularity.
        - Public data comes with survivorship bias, platform coverage gaps, and neighborhood-level masking.
        """
    )
    st.caption(
        "The challenge is not a lack of neighborhoods — it is a lack of halal-specific decision support."
    )

    st.divider()

    top_left, top_right = st.columns([1.1, 0.9], gap="large")
    with top_left:
        st.markdown("### The Approach")
        st.markdown(
            """
            We combine three phases of analysis:

            - **Phase 1:** identify halal market type
            - **Phase 2:** rank neighborhood opportunity
            - **Phase 3:** add risk and forward-looking insight
            """
        )
    with top_right:
        st.metric("NYC NTAs", "195")
        st.metric("NTAs scored", "144")
    # Image loading handled at module level

    st.divider()

    st.markdown("### The Result")
    st.markdown(
        """
        The system produces a merchant-facing shortlist of NYC neighborhoods with:

        - ranked opportunity score
        - halal market type
        - risk bucket and confidence
        - similar neighborhoods
        - demand and entry insight
        """
    )
    st.caption("Designed as decision support, not as a guaranteed forecast.")

    st.info(
        "The final output is a ranked neighborhood shortlist for halal food merchants, with current opportunity as the main signal and risk/forecast shown as supporting context."
    )

    st.divider()

    st.subheader("Phase 1 — Halal Market Typing")
    st.write(
        "Each NYC Neighborhood Tabulation Area (NTA) is clustered into one of four "
        "halal market types using KMeans clustering (implemented from scratch in NumPy)."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Features used:**")
        st.markdown("- `demand_score` — halal Yelp review share (proxy)")
        st.markdown("- `latent_demand_score` — implicit halal interest from review labels, keyword density, and log-normalized volume")
        st.markdown("- `halal_supply_rate` — halal cuisine density (proxy)")
        st.markdown("- `halal_cuisine_diversity_norm` — normalized cuisine diversity")
    with col2:
        st.markdown("**Market types:**")
        st.markdown("- 🔴 High Opportunity — high demand, low supply")
        st.markdown("- 🔵 Established Hub — strong existing halal scene")
        st.markdown("- 🟢 Growing Market — moderate demand, little supply")
        st.markdown("- ⚫ Low Demand — limited halal activity")
    st.write("`latent_demand_score` captures implicit halal interest from review labels, keyword density in review text, and log-normalized review volume. It surfaces demand in neighborhoods without established halal restaurants.")
    st.caption("Silhouette score: 0.3963 · k=4 selected by elbow method")

    st.divider()

    st.subheader("Phase 2 — Opportunity Scoring and Ranking")
    st.write(
        "Each NTA receives a composite opportunity score combining three independent "
        "signals from different data sources."
    )
    st.markdown(
        "> `final_score = 0.4 × demand_score + 0.4 × gap_score + 0.2 × viability_score`"
    )
    st.markdown(
        "**Viability** is a rule-based index from NYC restaurant inspection data:"
    )
    st.markdown("- −1 point if critical violation rate > 75th percentile")
    st.markdown("- −1 point if Grade A rate < 25th percentile")
    st.write(
        "Each NTA also receives a list of similar neighborhoods computed using "
        "cosine similarity across all four feature dimensions."
    )
    st.caption("Spearman rank correlation (gap vs final score): 0.93")

    st.divider()

    st.subheader("Phase 3 — Risk & Insight Layer")
    st.write(
        "Phase 3 adds supplementary risk and demand signals. "
        "It does NOT change the main ranking — Phase 2 final_score drives recommendations."
    )
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**GMM Risk Clustering** (n=4 components)")
        st.markdown("- Soft-assigns each NTA to a risk environment")
        st.markdown("- `high_risk_prob` = probability of high/medium-high risk")
        st.markdown("- BIC supports n=4 · Silhouette: 0.40")
    with col4:
        st.markdown("**Ridge Demand Forecast**")
        st.markdown("- Uses 2022 features to project 2023 halal review share")
        st.markdown("- R² = 0.16 (comparable to persistence baseline)")
        st.markdown("- Treat as directional signal only")

    st.divider()

    st.subheader("Data Sources")
    st.markdown(
        "| Source | Purpose |\n"
        "|---|---|\n"
        "| Yelp reviews + Gemini labels | Halal demand signal (14,853 reviews, 144 NTAs) |\n"
        "| CAMIS / NYC DOHMH restaurant records | Halal supply signal (494 halal restaurants) |\n"
        "| NYC DOHMH inspection records | Viability and risk signals |\n"
        "| US Census / NTA boundaries | Geographic unit definition |"
    )

    st.divider()

    st.subheader("Important Caveats")
    st.warning(
        "All signals in this tool are proxies, not ground truth. "
        "Please interpret results with appropriate caution.",
        icon="⚠️",
    )
    st.markdown(
        "1. **`demand_score`** is a halal discussion proxy derived from Yelp review text, "
        "not a measurement of true consumer demand.\n"
        "2. **`halal_supply_rate`** is based on halal-relevant cuisine families "
        "(Pakistani, Middle Eastern, etc.), not certified halal restaurant counts.\n"
        "3. **`gap_score`** is a heuristic estimate of unmet demand, not an economic measurement.\n"
        "4. **Risk scores** reflect the general restaurant operating environment in each NTA, "
        "not halal-restaurant-specific risk.\n"
        "5. **`cluster_confidence`** below 0.25 means the neighborhood sits near a cluster boundary — its market type label is less certain."
    )
