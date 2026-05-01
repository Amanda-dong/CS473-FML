"""Presentation-style slide deck page for the halal pipeline project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "output"


def _render_intro_flowchart() -> None:
    svg = """
    <svg viewBox="0 0 980 1080" width="100%" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L8,3 z" fill="#BDBDBD"></path>
        </marker>
        <style>
          .title { fill: #F4F4F4; font: 700 18px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .subtitle { fill: #D7D7D7; font: 400 11px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .sourceTitle { fill: #F4F4F4; font: 700 14px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .sourceSub { fill: #D7D7D7; font: 400 11px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .algo { fill: #FFFFFF; font: 700 14px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .body { fill: #F0F0F0; font: 400 12px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .finalTitle { fill: #F4F4F4; font: 700 18px Helvetica, Arial, sans-serif; text-anchor: middle; }
          .finalSub { fill: #E6F5D4; font: 400 12px Helvetica, Arial, sans-serif; text-anchor: middle; }
        </style>
      </defs>

      <rect x="20" y="20" width="940" height="1040" rx="24" fill="#1F1F1F"/>

      <!-- Top data sources -->
      <rect x="60" y="50" width="220" height="86" rx="16" fill="#494943" stroke="#A8A8A8" stroke-width="1.5"/>
      <text x="170" y="86" class="sourceTitle">Yelp reviews</text>
      <text x="170" y="116" class="sourceSub">people mentioning halal</text>

      <rect x="380" y="50" width="220" height="86" rx="16" fill="#494943" stroke="#A8A8A8" stroke-width="1.5"/>
      <text x="490" y="86" class="sourceTitle">Restaurant records</text>
      <text x="490" y="116" class="sourceSub">which cuisines exist per area</text>

      <rect x="700" y="50" width="220" height="86" rx="16" fill="#494943" stroke="#A8A8A8" stroke-width="1.5"/>
      <text x="810" y="86" class="sourceTitle">Health inspections</text>
      <text x="810" y="116" class="sourceSub">grades and violation flags</text>

      <!-- Step 1 -->
      <rect x="180" y="190" width="520" height="124" rx="18" fill="#4A3FAE" stroke="#9188F5" stroke-width="1.5"/>
      <text x="440" y="228" class="title">Step 1 — what kind of market is this?</text>
      <text x="440" y="261" class="body">cluster neighborhoods by halal demand vs supply</text>
      <text x="440" y="288" class="body">output: High Opportunity / Established Hub / etc.</text>
      <text x="440" y="305" class="subtitle">silhouette 0.40 · k=4 selected by elbow method</text>
      <rect x="740" y="214" width="150" height="76" rx="16" fill="#4A3FAE" stroke="#9188F5" stroke-width="1.5"/>
      <text x="815" y="258" class="algo">KMeans</text>

      <!-- Step 2 -->
      <rect x="180" y="412" width="520" height="124" rx="18" fill="#0E6A57" stroke="#5FD1BC" stroke-width="1.5"/>
      <text x="440" y="450" class="title">Step 2 — which areas rank highest?</text>
      <text x="440" y="483" class="body">score = 40% demand + 40% gap + 20% safety</text>
      <text x="440" y="510" class="body">find similar neighborhoods for each area</text>
      <text x="440" y="527" class="subtitle">Spearman 0.93 · this score is the main ranking · 144 NTAs covered</text>
      <rect x="740" y="436" width="150" height="76" rx="16" fill="#0E6A57" stroke="#5FD1BC" stroke-width="1.5"/>
      <text x="815" y="480" class="algo">Cosine KNN</text>

      <!-- Step 3 -->
      <rect x="180" y="634" width="520" height="124" rx="18" fill="#8A3A17" stroke="#E59C73" stroke-width="1.5"/>
      <text x="440" y="672" class="title">Step 3 — is this area risky?</text>
      <text x="440" y="705" class="body">group areas by risk level · predict if halal demand will grow</text>
      <text x="440" y="732" class="body">extra context only</text>
      <text x="440" y="749" class="subtitle">R² = 0.16 · shown as supplementary insight · not the main score</text>
      <rect x="740" y="658" width="150" height="76" rx="16" fill="#8A3A17" stroke="#E59C73" stroke-width="1.5"/>
      <text x="815" y="690" class="algo">GMM +</text>
      <text x="815" y="714" class="algo">Regression</text>

      <!-- Final output -->
      <rect x="120" y="876" width="640" height="118" rx="18" fill="#2F6B0A" stroke="#8FD655" stroke-width="1.5"/>
      <text x="440" y="922" class="finalTitle">Top neighborhoods to open a halal restaurant</text>
      <text x="440" y="956" class="finalSub">ranked list · opportunity score · market type · risk level · similar areas</text>

      <!-- Arrows -->
      <line x1="170" y1="136" x2="320" y2="190" stroke="#BDBDBD" stroke-width="2.2" marker-end="url(#arrow)"/>
      <line x1="490" y1="136" x2="490" y2="190" stroke="#BDBDBD" stroke-width="2.2" marker-end="url(#arrow)"/>
      <line x1="810" y1="136" x2="810" y2="474" stroke="#BDBDBD" stroke-width="2.2" marker-end="url(#arrow)"/>

      <line x1="440" y1="314" x2="440" y2="412" stroke="#BDBDBD" stroke-width="2.2" marker-end="url(#arrow)"/>
      <line x1="440" y1="536" x2="440" y2="634" stroke="#BDBDBD" stroke-width="2.2" marker-end="url(#arrow)"/>
      <line x1="440" y1="758" x2="440" y2="876" stroke="#BDBDBD" stroke-width="2.2" marker-end="url(#arrow)"/>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_phase1_results() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "phase1_cluster_assignments.csv")


@st.cache_data(show_spinner=False)
def _load_phase2_results() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "phase2_opportunity_scores.csv")


@st.cache_data(show_spinner=False)
def _load_phase3_results() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "final_recommendations.csv")


def main() -> None:
    st.set_page_config(page_title="Presentation", page_icon="🖼️", layout="wide")

    slides = st.tabs(
        [
            "01 · Intro",
            "02 · Phase 1 — KMeans",
            "03 · Phase 2 — Cosine KNN",
            "04 · Phase 3 — GMM + Regression",
            "05 · Demo",
            "06 · References",
        ]
    )

    with slides[0]:
        st.markdown("## 🕌 NYC Halal Restaurant Opportunity Finder")
        st.markdown(
            "### Finding the best neighborhoods to open a halal restaurant in NYC"
        )
        st.divider()
        st.markdown("### Problem")
        st.markdown(
            "- Halal food merchants in NYC lack data-driven tools for location decisions"
        )
        st.markdown(
            "- Existing platforms give snapshots of current market conditions, not halal-specific opportunity signals"
        )
        st.markdown(
            "- No tool combines halal demand, supply gap, and business risk in one place"
        )

        st.divider()

        st.markdown("### Our Approach")
        st.markdown("- Built a 3-phase ML pipeline using real NYC open data")
        st.markdown(
            "- Phase 1: cluster 144 neighborhoods into halal market types using KMeans from scratch"
        )
        st.markdown(
            "- Phase 2: score and rank every neighborhood by opportunity using Cosine KNN"
        )
        st.markdown(
            "- Phase 3: layer on risk assessment and demand forecasting using GMM and Ridge Regression"
        )
        _render_intro_flowchart()
        st.divider()
        st.markdown("### Data Sources")
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.markdown("**Yelp + Gemini Labeling**")
            st.markdown("- review text labeled for halal mentions")
            st.markdown("- used to build the demand signal")
        with col2:
            st.markdown("**CAMIS restaurant records**")
            st.markdown("- cuisine type per restaurant")
            st.markdown("- used to build the supply signal")
        with col3:
            st.markdown("**Health inspections**")
            st.markdown("- grade and violation flags")
            st.markdown("- used to build the risk signal")
        st.divider()

    with slides[1]:
        st.markdown("## Phase 1 — KMeans")
        st.divider()
        left, right = st.columns([1.8, 1], gap="large")
        with left:
            st.markdown("**The Problem**")
            st.markdown(
                "- NYC has 144 NTAs in our dataset\n"
                "- We need to identify which ones represent similar halal market conditions"
            )
            st.markdown("**How It Works**")
            st.markdown(
                "- Each NTA is represented by 3 halal features\n"
                "- KMeans picks k starting points\n"
                "- Each NTA is assigned to its nearest center\n"
                "- Centers move and the process repeats until stable\n"
                "- Implemented from scratch using numpy only"
            )
            st.markdown("**Why We Chose This**")
            st.markdown(
                "- Hard cluster assignments create interpretable market type labels\n"
                "- Elbow method selected k=4 at the point where extra clusters stop improving separation"
            )
        with right:
            st.metric("Optimal k", "4")
            st.metric("NTAs Clustered", "144")
            st.metric("Inertia Drop at k=4", "46.9%")
        st.divider()
        st.markdown("### Results")
        phase1_df = _load_phase1_results()
        p1_left, p1_right = st.columns(2, gap="large")
        with p1_left:
            cluster_counts = phase1_df["market_type"].value_counts()
            st.bar_chart(cluster_counts)
        with p1_right:
            cluster_means = (
                phase1_df.groupby("market_type")[
                    ["demand_score", "halal_supply_rate", "gap_score"]
                ]
                .mean()
                .round(3)
                .reset_index()
            )
            st.dataframe(cluster_means, use_container_width=True, hide_index=True)
        st.markdown("**How to interpret the four market types**")
        st.markdown(
            "- **High Opportunity**: high halal demand, low existing halal supply, and the largest opportunity gap"
        )
        st.markdown(
            "- **Established Hub**: strong halal demand with visible existing halal supply, meaning the market is already active and mature"
        )
        st.markdown(
            "- **Growing Market**: moderate demand and still-limited supply, suggesting room for expansion but a weaker signal than High Opportunity"
        )
        st.markdown(
            "- **Low Demand**: weak halal demand signal and limited market activity, making it the lowest-priority segment"
        )
        st.caption(
            "High Opportunity: 35 · Established Hub: 12 · Growing Market: 60 · Low Demand: 37"
        )
        st.caption(
            "High Opportunity (35 NTAs) = highest halal demand, lowest supply — primary target for merchants"
        )

    with slides[2]:
        st.markdown("## Phase 2 — Cosine KNN")
        st.divider()
        left, right = st.columns([1.8, 1], gap="large")
        with left:
            st.markdown("**The Problem**")
            st.markdown(
                "- Score every neighborhood by halal opportunity\n"
                "- Identify which areas are most similar to each other"
            )
            st.markdown("**How It Works**")
            st.markdown(
                "- Each NTA is represented as a vector of 4 features\n"
                "- Cosine similarity measures the angle between vectors\n"
                "- Areas with the same halal profile point in the same direction regardless of size\n"
                "- The scoring formula weights demand and gap equally"
            )
            st.markdown("**Scoring Formula**")
            st.code(
                "final_score = 0.4 × demand + 0.4 × gap + 0.2 × viability",
                language="text",
            )
            st.markdown("**Why Cosine over Euclidean**")
            st.markdown(
                "- Cosine captures profile shape, not absolute magnitude\n"
                "- A small dense halal area and a large dense halal area are correctly identified as similar"
            )
        with right:
            st.metric("Spearman Correlation", "0.93")
            st.metric("NTAs Ranked", "144")
            st.metric("Similar neighbors per NTA", "3")
        st.divider()
        st.markdown("### Results")
        phase2_df = _load_phase2_results()
        top5 = (
            phase2_df.sort_values("final_score", ascending=False)[
                ["nta_id", "market_type", "final_score"]
            ]
            .head(5)
            .copy()
        )
        top5["final_score"] = top5["final_score"].round(3)
        st.dataframe(top5, use_container_width=True, hide_index=True)
        st.caption(
            "Gap score drives ranking as designed — system is internally consistent"
        )

    with slides[3]:
        st.markdown("## Phase 3 — GMM + Regression")
        st.divider()
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**GMM Risk Clustering**")
            st.markdown("**Problem**")
            st.markdown(
                "- Hard Low/Medium/High labels lose nuance\n"
                "- Some NTAs sit on the boundary between risk environments"
            )
            st.markdown("**How**")
            st.markdown(
                "- Model NTAs as drawn from 4 Gaussian distributions\n"
                "- Each NTA receives a probability of belonging to each risk group\n"
                "- Output is soft assignment, not a hard label"
            )
            st.markdown("**Why n=4**")
            st.markdown(
                "- BIC dropped from 517 at n=2 to 14 at n=4\n"
                "- The data strongly supports 4 components"
            )
        with col2:
            st.markdown("**Ridge Demand Forecast**")
            st.markdown("**Problem**")
            st.markdown("- Will halal demand in this area grow next year?")
            st.markdown("**How**")
            st.markdown(
                "- Use 2022 NTA features to predict 2023 halal review share\n"
                "- Ridge regularization helps control overfitting on the 71-NTA sample"
            )
            st.markdown("**Time separation**")
            st.markdown(
                "- 2022 features → 2023 target\n"
                "- This removes same-source leakage that affected earlier versions"
            )
            st.markdown("**Honest evaluation**")
            st.markdown(
                "- R² = 0.16\n"
                "- Comparable to the persistence baseline\n"
                "- Treated as directional insight, not a precise forecast"
            )
        st.divider()
        st.info("Phase 3 is supplementary — does not change the main ranking")
        st.markdown("### Results")
        phase3_df = _load_phase3_results()
        risk_counts = phase3_df["risk_bucket"].fillna("Unknown").value_counts()
        st.bar_chart(risk_counts)
        st.markdown("**Interpretation of the results**")
        st.markdown(
            "- The risk layer does not replace the opportunity ranking; it adds operating context.\n"
            "- High-opportunity areas can still carry medium or high risk, which helps merchants balance upside against execution difficulty.\n"
            "- In our top recommendations, areas like QN33 and BK58 remain attractive but are not risk-free, while lower-risk areas provide more conservative entry options."
        )
        st.caption(
            "Risk layer adds nuance without overriding the Phase 2 opportunity ranking"
        )

    with slides[4]:
        st.markdown("## Live Demo")
        st.divider()
        st.markdown("### What to show")
        st.markdown("- Filter by borough and market type")
        st.markdown("- Show Top 5 recommendation cards")
        st.markdown("- Expand Risk & Environment and Demand Insight panels")

    with slides[5]:
        st.markdown("## References")
        st.divider()
        st.markdown("### Dataset References")
        st.markdown(
            "**1. Yelp Open Dataset**  \n"
            "Yelp Inc. *Yelp Open Dataset*. Available at: "
            "https://business.yelp.com/data/resources/open-dataset/  \n"
            "Used for: restaurant review text, ratings, business locations"
        )
        st.markdown(
            "**2. Google Gemini API (halal labeling)**  \n"
            "Google DeepMind. *Gemini API*. Google LLC, 2024. Available at: "
            "https://ai.google.dev/  \n"
            "Used for: labeling Yelp reviews for halal relevance "
            "(explicit_halal, implicit_halal, not_related)"
        )
        st.markdown(
            "**3. DOHMH New York City Restaurant Inspection Results (CAMIS)**  \n"
            "NYC Department of Health and Mental Hygiene. "
            "*DOHMH New York City Restaurant Inspection Results*. NYC Open Data, updated daily. "
            "Available at: https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j  \n"
            "Used for: restaurant universe, cuisine types, inspection grades, violation flags"
        )
        st.markdown(
            "**4. 2020 Neighborhood Tabulation Areas (NTAs)**  \n"
            "NYC Department of City Planning. *2020 Neighborhood Tabulation Areas (NTAs)*. "
            "NYC Open Data, 2021. Available at: "
            "https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd  \n"
            "Used for: geographic unit definition, NTA boundary codes"
        )
        st.markdown(
            "**5. US Census American Community Survey (ACS)**  \n"
            "U.S. Census Bureau. *American Community Survey 5-Year Estimates*. Available at: "
            "https://www.census.gov/programs-surveys/acs  \n"
            "Used for: neighborhood-level demographic context"
        )
        st.divider()
        st.markdown("### Tools & Libraries")
        st.markdown("- scikit-learn — https://scikit-learn.org")
        st.markdown("- NumPy — https://numpy.org")
        st.markdown("- Streamlit — https://streamlit.io")
        st.markdown("- Pandas — https://pandas.pydata.org")
        st.markdown("- Plotly — https://plotly.com")


if __name__ == "__main__":
    main()
