# Design Document — NYC Halal Opportunity Finder

**Team:** Amanda Dong (`yd2825`), Tony Zhao (`sz3822`), Harsh Agarwal (`ha2957`), Siqi Zhu (`sz3950`), Catherine Yi (`cgy2014`)

## Repo structure

```
CS473-FML/
├── README.md                 # Orientation, datasets, pipeline commands, teammate links.
├── PROPOSAL.md               # Formal problem framing + methodological overview.
├── DESIGN.md                 # This document — layout, staffing, stubs.
├── requirements.txt          # Python packages for conda/venv reproducibility.
├── build_halal_scores.py    # Planned merge façade for layering scores once extended (currently dormant).
├── src/                     # Pure analytics modules reused by runners and tooling.
│   ├── halal_demand.py      # Joins Gemini-labeled Yelp text to NTAs; emits demand_score features.
│   ├── halal_opportunity.py # CAMIS cuisines → halal supply rates, gaps, diversification stats.
│   ├── halal_kmeans.py      # Custom NumPy k-means clustering + centroid labeling helpers.
│   ├── halal_similarity.py # Cosine-sim neighborhood look-alikes atop normalized vectors.
│   ├── halal_risk.py       # Inspection aggregation, viability heuristic, Gaussian-mixture risk layer.
│   └── halal_forecast.py    # Temporal ridge models for halal chatter + entry dynamics.
├── scripts/                 # Executable phases + diagnostics runnable from repo root.
│   ├── run_phase1.py       # Builds phase-1 elbows, clustering exports, centroid tables.
│   ├── run_phase2.py       # Merges risk viability, composites final_score + similarity rankings.
│   ├── run_phase3.py       # Fuses ridge forecasts, GMM risk, writes final CSV outputs.
│   └── check_camis_time.py # Sanity checks aligning CAMIS timelines with Yelp pivots & parquet QA.
├── data/
│   ├── raw/                 # Curated-but-large inputs (tracked per .gitignore where needed).
│   └── processed/           # Downsized parquet features (inspection aggregates, crosswalk caches).
├── data/output/             # Derived CSVs powering Streamlit dashboards (clusters, finals, QC).
└── frontend/                # Multi-page Streamlit client + reusable UI widgets.
    ├── app.py               # Primary recommender UX (loads final_recommendations.csv).
    ├── methodology_content.py # Narrative + metrics supporting investor-style storytelling slides.
    ├── pages/               # Extra Streamlit surfaces (e.g. presentation deck).
    └── components/         # Sidebar filters, Plotly/NYC map stubs, expandable cards.

```

Notes: Paths reference the working codebase; placeholders called out explicitly below exist for modules still being generalized (e.g., multi-cuisine overlays).

## Algorithms used in `main` branch

Current `main` uses a 3-phase analytics pipeline:

1. **Phase 1 — Demand / Supply / Clustering**
   - Demand signal extraction from Yelp + Gemini labels (`src/halal_demand.py`)
   - Supply-gap feature construction from CAMIS-style data (`src/halal_opportunity.py`)
   - Unsupervised neighborhood clustering via custom NumPy k-means (`src/halal_kmeans.py`)

2. **Phase 2 — Ranking / Similarity / Viability**
   - Cosine-similarity neighborhood retrieval (`src/halal_similarity.py`)
   - Composite opportunity scoring and ranking in phase runners (`scripts/run_phase2.py`)
   - Inspection-derived viability features used for ranking context (`src/halal_risk.py`)

3. **Phase 3 — Risk / Forecast / Final score adjustment**
   - Gaussian Mixture Model (GMM) risk overlays (`src/halal_risk.py`)
   - Ridge-based forecasting for halal demand and entry trends (`src/halal_forecast.py`)
   - Final adjusted ranking output assembly (`scripts/run_phase3.py`)

Outputs from these phases are exported to `data/output/` and consumed by the
Streamlit frontend.


## Division of labor

| Teammate | Primary modules / artifacts |
| --- | --- |
| **Amanda Dong** | Owns Streamlit UX (`frontend/app.py`, reusable components under `frontend/components/*`), narrative assets in `frontend/methodology_content.py`, and ensures Plotly-backed map summaries stay synchronized with analytic outputs. |
| **Catherine Yi** | Leads probabilistic overlays and forecasting: Gaussian mixture workflows plus ridge/CV experimentation in `src/halal_forecast.py` / `src/halal_risk.py`, plus orchestration hygiene inside `scripts/run_phase3.py`. |
| **Harsh Agarwal** | Steward for CAMIS-aligned supply metrics (`src/halal_opportunity.py`), dataset contracts for `data/raw/*.csv`, and reproducibility tooling such as `scripts/check_camis_time.py` + documentation of hygiene extracts. |
| **Siqi Zhu** | Responsible for Gemini ↔ Yelp ingestion logic (`src/halal_demand.py`), labeled-review data QA under `data/raw/`, and documentation of demand-signal assumptions. |
| **Tony Zhao** | Coaches unsupervised structuring and ranking blends: executes `HalalKMeans` + elbows via `scripts/run_phase1.py`, composites Stage-2 rankings + cosine profiling in `src/halal_similarity.py`/`scripts/run_phase2.py`, and curates centroid interpretation tables surfaced in decks (`frontend/pages/presentation.py`). |


