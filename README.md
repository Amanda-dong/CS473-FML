# NYC Halal Market Intelligence & Opportunity Engine

## Data Availability

Due to GitHub file size limits and dataset licensing constraints, several large raw datasets are not stored directly in the repository. Please retrieve them via the repository Releases section.

---

## Pivot Note (read first)

This project moved from [`main-pre-pivot`](https://github.com/Amanda-dong/CS473-FML/tree/main-pre-pivot) to the current [`main`](https://github.com/Amanda-dong/CS473-FML/tree/main) implementation.

**Why we changed:**
In the integrated feature pipeline, missing-value pressure was high in several joins, which increased fallback/imputation usage (including median-based fills). That made some outputs less reliable for decision-facing recommendations. The current branch prioritizes realistic model-facing behavior and clearer output interpretation.

**What `main-pre-pivot` used (simple summary):**
- Broader integrated ETL/feature datasets across multiple NYC sources (see [pre-pivot design doc](https://github.com/Amanda-dong/CS473-FML/blob/main-pre-pivot/docs/Design.md) for the full list)
- Full ML stack including trajectory clustering (k-means / GMM), survival modeling (Cox PH + Random Survival Forest), learned scoring (XGBoost), ranking (LambdaMART), and explainability modules

**What we gained from the pivot:**
- Elimination of cascading imputation chains that obscured signal provenance
- Tighter control over each scoring component — every number in the final output is traceable to a documented formula
- A cleaner separation between uncertainty (Bayesian credible intervals) and missing data, rather than blending them via median fills

**What we still reuse from pre-pivot:**
We explicitly reuse partial datasets, especially `data/raw/gemini_labels_full.csv`, `data/raw/yelp_reviews_with_zones.csv`, and `data/processed/inspections.parquet`. The code that generates these reused datasets lives in the `main-pre-pivot` branch. We do not claim full algorithm reuse — the current branch uses a different, purpose-built `halal_*` phase pipeline (described in [DESIGN.md](DESIGN.md)).

**Branch references:**
- [main](https://github.com/Amanda-dong/CS473-FML/tree/main) — current implementation
- [main-pre-pivot](https://github.com/Amanda-dong/CS473-FML/tree/main-pre-pivot) — prior full-stack approach
- [Pre-pivot design doc](https://github.com/Amanda-dong/CS473-FML/blob/main-pre-pivot/docs/Design.md)

---

## Technical Abstract

Our engine implements an advanced **Multi-Signal Bayesian Fusion** pipeline that resolves two fundamental problems in urban market analysis: circular demand bias (where low-supply neighborhoods appear low-demand) and spatial autocorrelation in opportunity rankings. By synthesizing latent demand signals from LLM-labeled review corpora, GMM-based probabilistic risk quantification, and spatial econometrics (Local Moran's I), the system distinguishes between perceived market saturation and genuine, structurally underserved opportunities.

This research-grade framework delivers neighborhood-level predictive intelligence with rigorously quantified uncertainty — moving beyond legacy supply-gap heuristics toward a statistically principled methodology for high-stakes urban site selection.

---

## Analytical Pipeline Architecture

The engine operates in three distinct phases, each a discrete transformation of the feature space, converging on a risk-adjusted, spatially-aware opportunity ranking.

| Phase | Methodology | Primary Output |
|-------|-------------|----------------|
| **Phase 1: Market Characterization** | Bayesian Demand Extraction (Beta conjugate priors) + Latent Signal Fusion + k-means++ Clustering | Unsupervised market segments with cluster confidence scores |
| **Phase 2: Contextual Retrieval** | Cosine Similarity Profiling + Multi-criteria Composite Scoring + LISA spatial integration | NTA-to-NTA look-alike clusters and initial opportunity rankings |
| **Phase 3: Risk & Forecasting** | GMM Risk Overlay (BIC-selected) + RidgeCV Demand Forecasting + Confidence-Adjusted Rank Fusion | Final viability scores with probabilistic risk buckets and temporal growth signals |

---

## Key Technical Pillars

### 1. Bayesian Demand Signal Modeling

To mitigate stochastic noise in low-traffic NTAs, we employ **Bayesian shrinkage** using Beta conjugate priors on observed halal review share.

- **Shrinkage Formula**: `shrunk_share = (halal_count + α × global_mean) / (total + α)` with prior strength α = 10
- **Time-Decay Weighting**: Review signals are weighted by `ω = 0.85^Δt` (years elapsed) to prioritize recent market shifts
- **Population Normalization**: Demand is expressed as halal mentions per 1,000 residents, correcting for neighborhood density variation
- **80% Credible Intervals**: Per-NTA uncertainty bounds via `scipy.stats.beta.ppf`, enabling merchants to distinguish high-signal from high-noise opportunities

### 2. Latent Demand Signal Extraction

We decouple demand from supply to eliminate circular measurement bias. The latent demand signal fuses three orthogonal proxies:

- **Implicit LLM Labels**: Gemini zero-shot classification of Yelp reviews (explicit_halal, implicit_halal, not_related)
- **Keyword Density**: Presence of halal-adjacent vocabulary in review discourse independent of branded merchants
- **Activity Signal**: Review volume and recency as a proxy for neighborhood culinary engagement

This signal is estimated independently of existing halal merchant presence — capturing "hidden" demand in neighborhoods that currently lack a formal halal footprint.

### 3. Unsupervised Market Segmentation (k-means++)

A custom **k-means++ implementation** segments NYC's 260+ NTAs into four market profiles. The k-means++ initialization uses distance-proportional seeding (vectorized via NumPy broadcast) to guarantee global convergence:

- **Cluster Confidence Score**: Centroid separation ratio flags borderline NTAs, providing a decision uncertainty index
- **Features**: Demand score, latent demand, halal supply rate, and cuisine diversity (normalized)
- **Model Selection**: Elbow + silhouette analysis across k ∈ [2, 8] for optimal segment count

### 4. Spatial Market Intelligence (LISA)

Neighborhoods are modeled as a continuous spatial field, not isolated data points. **Local Moran's I (LISA)** identifies:

- **Hot Spots (HH)**: Clusters of high halal demand and density — saturated markets
- **Underserved Zones (LH)**: Low-supply NTAs surrounded by high-demand neighbors — primary targets for entry; receive an 8% score boost
- **Spatial Outliers (HL)**: Isolated high-supply islands in low-demand regions — fragile markets
- Statistical significance filtered at p < 0.05 (permutation test, 999 iterations)

### 5. Probabilistic Risk Assessment (GMM)

**Gaussian Mixture Models** with **BIC-selected components** cluster neighborhoods across a multidimensional risk surface (critical violation rate, grade-A compliance rate, inspection frequency, demand score). This yields:

- A continuous `high_risk_prob` score rather than binary classification
- `risk_bucket` (Low / Medium / High) derived from GMM component probabilities
- Silhouette-validated cluster separation for reliability assurance

### 6. Predictive Growth Forecasting (RidgeCV)

Two **RidgeCV** models (alpha ∈ [0.001, 100], 5-fold KFold CV) provide temporal intelligence:

- **Demand Forecast**: Predicts 2024 halal-related review share from 2022–2023 NTA signals
- **Entry Forecast**: Predicts new halal merchant openings from review momentum and supply features
- Both models include ablation tables, persistence baselines, and in-sample R² diagnostics

### 7. Explainable Recommendations (SHAP-style Decomposition)

Each recommendation card provides a **linear score decomposition** — the contribution of demand signal (40%), supply gap (40%), and viability (20%) to the final score, rendered as a Plotly horizontal bar chart. This provides merchant-facing transparency into exactly why an NTA ranked where it did.

---

## Dashboard Features

The interactive Streamlit dashboard (`streamlit run frontend/app.py`) provides three analytical views:

### Tab 1 — Map & Shortlist
- **MapBox opportunity map**: NTAs colored by market type and score intensity
- **Top-3 quick summary**: Score gauge, market type badge, adjusted score
- **Recommendation cards**: Radar chart (5 dimensions), score gauge, plain-English summary, SHAP decomposition, review evidence, risk detail, and growth outlook

### Tab 2 — Side-by-Side Comparison
- **Portfolio analysis tool**: Select any two neighborhoods from your shortlist for head-to-head evaluation
- **Dual radar charts**: Direct visual comparison of opportunity profiles
- **Delta metrics**: Quantified advantage in demand, gap, and overall fit
- **Narrative synthesis**: Auto-generated plain-English verdict on which neighborhood leads in which dimension

### Tab 3 — Market Analytics
- **Score distribution**: Box plot across all four market segments
- **Demand vs. gap scatter**: Strategic positioning of every NTA in opportunity space
- **Full comparison table**: Sortable, filterable view of the complete model output

---

## Evaluation & Validation

| Metric | Purpose |
|--------|---------|
| Silhouette Score | Cluster separation quality (k-means++) |
| BIC | GMM component count selection |
| In-sample R² + Persistence Baseline | RidgeCV forecasting reliability |
| Moran's I (p-value) | Spatial signal significance |
| 80% Credible Interval Width | Per-NTA demand uncertainty |

The pipeline includes a comprehensive **validation framework** (`tests/`) covering unit tests for all core modules, and `scripts/run_all.py` which orchestrates all three phases end-to-end with post-run column and NTA-count assertions.

---

## Near-Term Research Extensions

- **Spatial Autoregressive Models (SAR)**: Explicitly modeling demand spillover across neighboring NTAs using spatial lag operators — currently in prototyping
- **Dynamic GMM (Hidden Markov)**: Evolving the static risk model into a temporal state-transition model to capture hygiene trajectory, not just current snapshot
- **Neural Demand Embeddings**: Fine-tuned sentence transformers on the Yelp + Gemini label corpus to capture semantic halal interest profiles beyond keyword heuristics

---

## Team & Contributors

| Name             | NYU NetID | Role / Specialization |
|------------------|-----------|-----------------------|
| Amanda Dong      | `yd2825`  | UX Lead / Visualization Architecture |
| Tony Zhao        | `sz3822`  | Unsupervised Learning / Ranking Systems |
| Harsh Agarwal    | `ha2957`  | Data Engineering / Hygiene Pipelines |
| Siqi Zhu         | `sz3950`  | NLP / Demand Signal Processing |
| Catherine Yi     | `cgy2014` | Risk Modeling / Forecasting |

---

## Execution Guide

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager: `pip install uv`
- Raw datasets retrieved from the repository Releases section and placed in `data/raw/`

---

### One-Command Launch (Dashboard Only)

If you have the pre-computed model outputs in `data/output/` (included in the repo), launch the dashboard directly:

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt && streamlit run frontend/app.py
```

This installs all dependencies and opens the dashboard at `http://localhost:8501`.

---

### Full Reproducible Run (Pipeline + Dashboard)

To re-run the entire analytical pipeline from raw data and then launch the dashboard:

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt && python scripts/run_all.py && streamlit run frontend/app.py
```

`run_all.py` executes all three phases sequentially and validates output column integrity after each phase before the dashboard opens.

---

### Granular Execution

```bash
python scripts/run_phase1.py   # Market Characterization & Clustering
python scripts/run_phase2.py   # Contextual Retrieval & Scoring
python scripts/run_phase3.py   # Risk Overlays & Forecasting
```

### Tests

```bash
uv pip install pytest && pytest tests/ -v
```
