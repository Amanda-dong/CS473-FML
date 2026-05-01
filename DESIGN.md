# System Design & Algorithmic Framework

**Project:** NYC Halal Market Intelligence  
**Team:** Amanda Dong (`yd2825`), Tony Zhao (`sz3822`), Harsh Agarwal (`ha2957`), Siqi Zhu (`sz3950`), Catherine Yi (`cgy2014`)

## Architectural Overview

The system is designed as a decoupled, 3-phase analytic pipeline where each stage performs a discrete transformation of the feature space, ultimately converging on a risk-adjusted opportunity ranking.

```
CS473-FML/
├── src/                         # Core Algorithmic Engine
│   ├── halal_demand.py          # Bayesian demand extraction & shrinkage
│   ├── halal_opportunity.py     # Multi-cuisine supply-gap & diversification metrics
│   ├── halal_kmeans.py          # k-means++ unsupervised segmentation
│   ├── halal_similarity.py      # Cosine-based contextual retrieval
│   ├── halal_risk.py            # GMM-based probabilistic risk modeling
│   └── halal_forecast.py        # RidgeCV temporal growth prediction
├── scripts/                     # Phase Runners & Orchestration
│   ├── run_phase1.py            # Market Characterization Pipeline
│   ├── run_phase2.py            # Contextual Retrieval & Scoring Pipeline
│   └── run_phase3.py            # Risk-Adjusted Forecasting Pipeline
├── frontend/                    # Decision-Support Interface
│   ├── app.py                   # Streamlit UI with SHAP-style explainability
│   └── components/              # Interactive Plotly & MapBox modules
└── data/                        # Data Persistence Layer
    ├── raw/                     # Yelp (text), Gemini (labels), CAMIS (admin)
    ├── processed/               # Parquet-optimized hygiene & demographic features
    └── output/                  # Serialized analytic results
```

---

## Methodological Deep-Dive

### 1. Phase 1 — Market Characterization & Clustering
This phase focuses on identifying the "ground truth" of the current market landscape.
- **Demand Estimation**: We utilize **Bayesian Shrinkage** (Beta-Binomial conjugate priors) to estimate the true share of halal-related demand in an NTA. This handles the variance in review counts by pulling low-volume NTAs toward the global mean.
- **Supply-Gap Analysis**: A comparison of explicit halal restaurant density (derived from CAMIS cuisine tagging) against the latent demand signal derived from Yelp text.
- **Segmentation**: A custom **k-means++** implementation segments the market into four tiers (e.g., *Established Hubs* vs. *High-Opportunity Gaps*) based on standardized demand, supply, and diversification vectors.

### 2. Phase 2 — Contextual Retrieval & Composite Ranking
Phase 2 shifts from global segmentation to local similarity and viability.
- **Cosine Profiling**: For every NTA, we calculate a similarity matrix to find "look-alike" neighborhoods across the feature space, facilitating peer-group benchmarking.
- **Viability Heuristics**: An initial rank is established using a composite weighted score of Demand, Supply Gap, and Inspection Viability.
- **Spatial Consistency (Near-term)**: We utilize **Local Moran's I (LISA)** to detect if a high-scoring NTA is a "Hot Spot" (part of a spatial cluster of opportunity) or a "Spatial Outlier".

### 3. Phase 3 — Probabilistic Risk & Temporal Forecasting
The final phase applies advanced ML overlays to ensure the stability of the recommendations.
- **Probabilistic Risk (GMM)**: Rather than simple thresholds, we fit a **Gaussian Mixture Model** (optimized via **Bayesian Information Criterion**) to capture the underlying distribution of restaurant hygiene risk. This allows for a "High Risk Probability" score that penalizes recommendations with unstable inspection trajectories.
- **Predictive Growth (RidgeCV)**: We implement two **RidgeCV** models (with cross-validated alpha selection) to forecast:
    1. Future halal demand share based on historical latent signals.
    2. Expected merchant entry rates (new restaurant registrations).
- **Final Rank Adjustment**: The final `final_score_adjusted` is a product of the base opportunity score, a risk penalty (derived from GMM), and a growth boost (derived from RidgeCV).

---

## Division of Labor & Module Ownership

| Specialist | Primary Domain & Module Responsibility |
| :--- | :--- |
| **Amanda Dong** | **UX Architecture & Visualization**: Leads the design of the Streamlit interface (`frontend/app.py`), focusing on the translation of complex ML scores into intuitive Plotly radar charts and MapBox layers. |
| **Catherine Yi** | **Probabilistic Modeling & Forecasting**: Architect of the Phase 3 pipeline. Owns the GMM risk implementation (`src/halal_risk.py`) and the RidgeCV growth models (`src/halal_forecast.py`). |
| **Harsh Agarwal** | **Data Engineering & Hygiene Quality**: Steward of the administrative supply records and inspection aggregates. Owns `src/halal_opportunity.py` and ensure temporal alignment via `scripts/check_camis_time.py`. |
| **Siqi Zhu** | **NLP & Demand Signal Engineering**: Owns the Yelp ↔ Gemini ingestion logic (`src/halal_demand.py`). Responsible for the Bayesian shrinkage implementation and latent signal extraction. |
| **Tony Zhao** | **Unsupervised Learning & Ranking Systems**: Executes the k-means++ segmentation logic (`src/halal_kmeans.py`) and the Phase 2 similarity/ranking engine (`src/halal_similarity.py`). |
