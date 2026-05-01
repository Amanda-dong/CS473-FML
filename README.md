# NYC Halal Market Intelligence & Opportunity Engine

## Data Availability
Due to GitHub file size limits and dataset licensing constraints, several large raw datasets are not stored directly in the repository. Please retrieve them via the repository Releases section.

---

## Technical Abstract

Our engine implements an advanced **Multi-Signal Bayesian Fusion** pipeline, solving circular demand bias and spatial autocorrelation in urban market analysis. By synthesizing latent demand signals, GMM-based risk quantification, and spatial econometrics (LISA), the system distinguishes between perceived market saturation and genuine, underserved opportunities. This research-grade framework provides neighborhood-level predictive intelligence with rigorously quantified uncertainty, moving beyond legacy supply-gap models to a future-proof methodology for high-stakes urban site selection.

---

## Analytical Pipeline Architecture

The engine operates in three distinct phases, progressively refining the recommendation set from latent signal extraction to actionable spatial strategy.

| Phase | Methodology | Primary Output |
|-------|-------------|----------------|
| **Phase 1: Market Characterization** | Bayesian Demand Extraction + Supply Gap + **Latent Signal + Spatial LISA integration** | Unsupervised market segments (Established Hubs, Growing Markets, High Opportunity) |
| **Phase 2: Contextual Retrieval** | Cosine Similarity Profiling + Multi-criteria Composite Scoring | NTA-to-NTA look-alike clusters and initial opportunity rankings |
| **Phase 3: Risk & Forecasting** | **GMM** Risk Overlay + **RidgeCV** Demand Forecasting + Final Rank Adjustment | Adjusted viability scores with probabilistic risk buckets and temporal growth signals |

---

## Key Technical Pillars

### 1. Latent Demand Signal Modeling
We decouple demand from existing supply to solve the circular demand bias (where low-supply neighborhoods show artificially low halal interest). 
- **Implicit Signal Fusion**: We aggregate implicit halal labels derived from LLM-processed reviews, keyword density across NTA discourse, and spatial activity signals.
- **Independence**: This signal is modeled independently of existing halal merchants, capturing "hidden" hunger for halal products in neighborhoods that currently lack a recognized footprint.

### 2. Cluster Confidence Scoring
To ensure robust decision-making, we quantify the reliability of neighborhood assignments using a **centroid separation ratio**. 
- **Borderline NTA Detection**: Neighborhoods with low separation scores are flagged, providing an uncertainty index that allows merchants to distinguish between "clear-cut" opportunities and high-variance markets.

### 3. Spatial Market Intelligence (LISA)
We treat neighborhoods not as isolated data points, but as a continuous field using **Local Moran's I (LISA)**.
- **Hot Spot Analysis**: Identification of statistically significant clusters of high halal demand.
- **Underserved Identification**: Specifically isolating **Low-High** neighborhoods—underserved NTAs surrounded by high-demand, high-consumption neighbor zones, signaling high spillover potential.

### 4. Probabilistic Risk Assessment (GMM)
We utilize **Gaussian Mixture Models (GMM)** with **BIC-selected components** to cluster neighborhoods based on multidimensional risk profiles. This provides a continuous probability density of "merchant viability risk," capturing subtle shifts in inspection frequency and DOHMH compliance.

### 5. Bayesian Demand Forecasting
Using **Bayesian shrinkage** via Beta conjugate priors, we handle the "cold-start" problem in low-traffic NTAs. We generate **80% Credible Intervals** for halal demand share, ensuring that our recommendations are supported by statistically significant evidence rather than stochastic noise.

---

## Results Highlights
- **144 NTAs Analyzed**: Comprehensive coverage across the NYC landscape.
- **MN22 Spotlight**: The Washington Square/NYU area was correctly identified as a high-potential market. Despite lower *revealed* historical supply, our latent demand model identified intense, unmet demand signals, validating the efficacy of the Bayesian-LISA fusion approach.
- **Market Breakdown**: 36% of analyzed NTAs categorized as "Emerging High-Opportunity."

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

### Full Pipeline (Recommended)
```bash
python scripts/run_all.py
```

### Granular Execution
```bash
# Phase 1: Market Characterization
python scripts/run_phase1.py
# Phase 2: Contextual Retrieval
python scripts/run_phase2.py
# Phase 3: Risk & Forecasting
python scripts/run_phase3.py
```

### Dashboard Deployment
```bash
streamlit run frontend/app.py
```
