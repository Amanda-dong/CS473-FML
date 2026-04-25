from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_model_info() -> dict:
    info = {"survival": {}, "ranking": {}, "scoring": {}}
    models_dir = Path("data/models")
    for name in ("survival_model", "ranking_model", "scoring_model"):
        path = models_dir / f"{name}.joblib"
        if path.exists():
            try:
                bundle = joblib.load(path)
                info[name.split("_")[0]] = {
                    "path": str(path),
                    "keys": list(bundle.keys()) if isinstance(bundle, dict) else [type(bundle).__name__],
                }
            except Exception as e:
                info[name.split("_")[0]] = {"error": str(e)}
    return info


def render_methodology_page() -> None:
    st.header("📖 Methodology")
    st.markdown(
        """
        This page documents how the NYC Healthy-Food White-Space Finder scores and ranks
        micro-zones for new quick-service healthy-food concepts, from raw data ingestion
        through opportunity ranking, survival-risk estimation, and trajectory clustering.
        """
    )

    # Section 1: Problem Framing
    st.subheader("1. Problem Framing")
    st.markdown(
        """
        We operate at the granularity of NYC **micro-zones** — Neighborhood Tabulation
        Areas (NTAs) — and look for places where a healthy-food concept has the strongest
        combination of underserved demand, viable merchant economics, and low survival
        risk. Each zone is classified into one of four zone types:

        - **Campus walkshed** — high student foot traffic around universities.
        - **Lunch corridor** — dense daytime-worker catchments with quick-service demand.
        - **Transit catchment** — strong commuter flow around rail / multimodal hubs.
        - **Business district** — office-heavy cores with weekday lunch peaks.
        """
    )

    # Section 2: Data Sources
    st.subheader("2. Data Sources")
    st.markdown(
        "Sources are organized into two tiers: **Tier 1 (Core)** drives the CMF score "
        "directly; **Tier 2 (Enrichment)** refines signals and adds qualitative context."
    )
    sources = pd.DataFrame(
        [
            {"Tier": "Tier 1 (Core)", "Source": "NYC Open Data — Restaurant Inspections (DOHMH)", "Use": "Healthy vs. non-healthy establishment counts and subtype gap analysis"},
            {"Tier": "Tier 1 (Core)", "Source": "NYC Open Data — DOB Permits", "Use": "Merchant license velocity and turnover signal"},
            {"Tier": "Tier 1 (Core)", "Source": "Citi Bike trip data", "Use": "Micro-mobility proxy for foot traffic and transit catchment strength"},
            {"Tier": "Tier 1 (Core)", "Source": "U.S. Census ACS 5-year", "Use": "Income gradient and demographic weighting"},
            {"Tier": "Tier 1 (Core)", "Source": "NYC NTA boundary shapefile", "Use": "Micro-zone geometry and spatial joins"},
            {"Tier": "Tier 2 (Enrichment)", "Source": "Yelp Fusion API", "Use": "Reviews, ratings, and cuisine categories for merchant viability"},
            {"Tier": "Tier 2 (Enrichment)", "Source": "Inside Airbnb", "Use": "Listing density as a transient / visitor-flow proxy"},
        ]
    )
    st.dataframe(sources, hide_index=True, use_container_width=True)

    # Section 3: Zone Typing
    st.subheader("3. Zone Typing")
    st.markdown(
        """
        Each NTA is classified into its zone type using ACS demographics (age mix,
        educational attainment, household composition), employment density from permit
        and business-activity data, and transit / POI proximity. The resulting zone type
        conditions downstream demand modeling — a campus walkshed weights student
        lunch patterns differently from a business district's weekday peak.
        """
    )

    # Section 4: CMF Opportunity Score
    st.subheader("4. CMF Opportunity Score")
    st.markdown(
        """
        The **Composite Micro-zone Fitness (CMF)** score is a weighted sum of ten
        normalized signals in [0, 1]. Base signals add to the score; penalty signals
        (competition saturation, rent pressure) subtract from it. Weights below are the
        production values from `src/models/cmf_score.py`.
        """
    )
    weights = pd.DataFrame(
        [
            {"Signal": "Quick-lunch demand", "Weight": 0.20, "Type": "Base"},
            {"Signal": "Merchant viability", "Weight": 0.18, "Type": "Base"},
            {"Signal": "Subtype healthy gap", "Weight": 0.16, "Type": "Base"},
            {"Signal": "General healthy gap", "Weight": 0.12, "Type": "Base"},
            {"Signal": "License velocity", "Weight": 0.10, "Type": "Base"},
            {"Signal": "Review volume", "Weight": 0.08, "Type": "Base"},
            {"Signal": "Transit proximity", "Weight": 0.07, "Type": "Base"},
            {"Signal": "Income gradient", "Weight": 0.05, "Type": "Base"},
            {"Signal": "Competition penalty", "Weight": 0.08, "Type": "Penalty"},
            {"Signal": "Rent penalty", "Weight": 0.04, "Type": "Penalty"},
        ]
    )
    st.dataframe(weights, hide_index=True, use_container_width=True)

    # Section 5: Survival Risk
    st.subheader("5. Survival Risk")
    st.markdown(
        """
        Survival risk estimates the probability that a new merchant in the zone fails to
        reach a 12-month operating milestone. The production model is a **Cox
        Proportional Hazards / XGBoost hybrid**: Cox provides calibrated hazard ratios
        over tenure, and XGBoost captures non-linear interactions among zone features.
        When the learned model is unavailable or falls back, we surface a heuristic
        proxy of **(1 − merchant_viability_score)** so the UI always has a defensible
        risk estimate.
        """
    )

    # Section 6: Zone Trajectory Clustering
    st.subheader("6. Zone Trajectory Clustering")
    st.markdown(
        """
        Zones are grouped by trajectory using **K-Means** over a time-windowed feature
        vector (license velocity deltas, review-volume growth, demographic drift). The
        resulting clusters map to four trajectory labels — **emerging**, **gentrifying**,
        **stable**, and **declining** — and are displayed on each recommendation card as
        a trajectory badge.
        """
    )

    # Section 7: Model Configuration
    with st.expander("7. Model Configuration"):
        info = _load_model_info()
        st.json(info)

    st.caption(
        "Sources: NYC Open Data · U.S. Census · Yelp · Inside Airbnb · Citi Bike."
    )
