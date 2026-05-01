# NYC Halal Market Intelligence & Opportunity Engine

## Data Availability (Important)

Due to GitHub file size limits and dataset licensing constraints, several large raw datasets are not stored directly in the repository.

Instead, they are provided via the repository Releases section.

An advanced, data-driven framework for identifying high-potential neighborhood entry points for halal restaurant merchants. This project synthesizes large-scale unstructured review text (Yelp + LLM-based labeling), administrative supply records (CAMIS), and spatial hygiene data into a sophisticated 3-phase analytical pipeline.

## Technical Abstract

Our methodology moves beyond simple supply-demand gaps to incorporate **Bayesian demand estimation**, **unsupervised market segmentation**, and **probabilistic risk overlays**. We utilize a hybrid approach combining frequentist time-series forecasting with Bayesian shrinkage to handle the "cold-start" problem in low-traffic Neighborhood Tabulation Areas (NTAs).

---

## Analytical Pipeline Architecture

The engine operates in three distinct phases, progressively refining the recommendation set from raw signal extraction to predictive viability.

| Phase | Methodology | Primary Output |
|-------|-------------|----------------|
| **Phase 1: Market Characterization** | Bayesian Demand Extraction + Supply Gap Analysis + **k-means++** Clustering | Unsupervised market segments (Established Hubs, Growing Markets, High Opportunity) |
| **Phase 2: Contextual Retrieval** | Cosine Similarity Profiling + Multi-criteria Composite Scoring | NTA-to-NTA look-alike clusters and initial opportunity rankings |
| **Phase 3: Risk & Forecasting** | **GMM** Risk Overlay + **RidgeCV** Demand Forecasting + Final Rank Adjustment | Adjusted viability scores with probabilistic risk buckets and temporal growth signals |

---

## Key Technical Pillars

### 1. Bayesian Demand Signal Modeling
To mitigate noise in low-volume NTAs, we employ **Bayesian shrinkage** using Beta conjugate priors.
- **Time-Decay Weighting**: Review signals are weighted by $\omega = 0.85^{\Delta t}$ (years) to prioritize recent market shifts.
- **Uncertainty Quantification**: We calculate **80% Credible Intervals** for halal demand share, allowing the system to distinguish between high-signal and high-noise opportunities.
- **Normalization**: Demand is normalized per-capita (halal mentions per 1,000 residents) to account for varying neighborhood densities.

### 2. Unsupervised Market Segmentation (k-means++)
Using a custom **k-means++** implementation, we segment NYC's 260+ NTAs into distinct market profiles. This initialization strategy ensures global convergence and stable cluster assignment for features including latent demand density and supply diversification.

### 3. Probabilistic Risk Assessment (GMM)
We utilize **Gaussian Mixture Models (GMM)** with **BIC-selected components** to cluster neighborhoods based on multidimensional risk profiles (DOHMH critical violation rates, inspection frequency, and grade stability). This provides a continuous probability density of "High Risk" rather than a binary classification.

### 4. Predictive Growth Forecasting
A **RidgeCV** implementation (alpha search across $[0.001, 100]$) forecasts future halal demand and merchant entry trends. This allows the model to "boost" neighborhoods that exhibit strong positive momentum in latent market chatter.

### 5. Explainable Recommendations (SHAP-style)
The final recommendation engine utilizes a **linear score decomposition** (inspired by SHAP values) to show users exactly how much demand, supply-gap, and risk contributed to a neighborhood's specific rank.

---

## Near-Term & Experimental Scope
- **Spatial Autocorrelation**: Integration of **Local Moran's I (LISA)** to detect spatial clusters of halal opportunity (Hot Spots) vs. isolated outliers.
- **Dynamic GMM**: Evolving the risk model into a Hidden Markov Model (HMM) to capture state-transitions in neighborhood hygiene quality.

---

## Team & Contributors

| Name             | NYU NetID | Role / Specialization |
|------------------|-----------|-----------------------|
| Amanda Dong      | `yd2825`  | UX Lead / Visualization Arch |
| Tony Zhao        | `sz3822`  | Unsupervised Learning / Ranking |
| Harsh Agarwal    | `ha2957`  | Data Engineering / Hygiene Pipelines |
| Siqi Zhu         | `sz3950`  | NLP / Demand Signal Processing |
| Catherine Yi     | `cgy2014` | Risk Modeling / Forecasting |

---

## Execution Guide

### Environment Bootstrap
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Reproduce Analytic Results
```bash
# Phase 1: Clustering & Supply-Gap
python scripts/run_phase1.py
# Phase 2: Similarity & Composite Scoring
python scripts/run_phase2.py
# Phase 3: Risk Overlays & Forecasting
python scripts/run_phase3.py
```

### Dashboard Deployment
```bash
streamlit run frontend/app.py
```
