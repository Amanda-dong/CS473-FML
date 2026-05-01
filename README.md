# NYC Restaurant Intelligence Platform

A shortlist-first decision tool that helps independent operators answer one
question: **where in NYC should a merchant open a healthier fast-casual
restaurant, and which underserved zones should they shortlist first?**

Updated: April 30, 2026
Course project · Spring 2026
Repo: <https://github.com/Amanda-dong/CS473-FML>

## Teammates

- Amanda Dong (`yd2825`) — [GitHub](https://github.com/Amanda-dong)
- Tony Zhao (`sz3822`) — [GitHub](https://github.com/Tonyzsp)
- Harsh Agarwal (`ha2957`) — [GitHub](https://github.com/harshagarwalnyu)
- Siqi Zhu (`sz3950`) — [GitHub](https://github.com/HelenZhutt)
- Catherine Yi (`cgy2014`) — [GitHub](https://github.com/catherinegyi)

---

## Table of Contents

1. [Problem & Product](#1-problem--product)
2. [Why This Version Is Different](#2-why-this-version-is-different)
3. [Recommended User Experience](#3-recommended-user-experience)
4. [Core ML Stack](#4-core-ml-stack)
5. [Data Strategy](#5-data-strategy)
6. [Repository Structure](#6-repository-structure)
7. [Division of Labor](#7-division-of-labor)
8. [Sprint Plan & Status](#8-sprint-plan--status)
9. [Setup](#9-setup)
10. [Quick Start](#10-quick-start)
11. [Repository Status & Artifacts](#11-repository-status--artifacts)
12. [Documentation Index](#12-documentation-index)

---

## 1. Problem & Product

Independent restaurant owners in NYC make high-cost neighborhood decisions
with far less market intelligence than chains. This project closes part of
that gap by producing a recommendation workflow that is decision-useful, not
another generic city dashboard:

- enter a healthy-food concept and a few constraints
- get a ranked shortlist of underserved zones
- see the evidence behind each recommendation
- understand the risk, confidence, and tradeoffs before signing a lease

Restaurant success is a **timing** problem as much as a location problem. We
operationalize that timing signal with a defensible, time-aware data pipeline
rather than a static neighborhood score, focusing on a distinctive merchant
problem: **finding healthy-food white space in dense urban micro-markets such
as campus-adjacent lunch corridors**.

Full problem framing and methodology rationale: [`docs/Proposal.md`](docs/Proposal.md).

## 2. Why This Version Is Different

- Official NYC data is the backbone. Third-party platforms are enrichment
  layers, not the system of record.
- Neighborhood phases are **discovered with unsupervised learning** instead of
  weak hand-labeled classes.
- Restaurant survival is modeled explicitly with **time-to-event methods**
  instead of forcing everything into classification.
- Review sentiment is bootstrapped with **Gemini-generated silver labels** and
  aggregated directly, avoiding GPU-heavy local fine-tuning in the main plan.
- The final product is **shortlist-first** and **explanation-first**, not
  map-first.
- The product has a distinctive use case: healthy-food white space near
  campuses, office clusters, and lunch corridors.

## 3. Recommended User Experience

1. User enters a healthy concept subtype (salad bowls, Mediterranean bowls,
   healthy Indian / South Asian bowls, protein-heavy lunch, vegan
   grab-and-go) plus optional constraints.
2. System returns the top 5 underserved micro-zones — not every geography at
   once.
3. Each recommendation card shows:
   - overall opportunity rank
   - confidence bucket
   - healthy supply-gap summary
   - recommended concept subtype for that zone
   - key positive drivers
   - key risk flags
   - similar existing restaurants
   - data freshness note
4. The user can switch into a scenario view to see what changes when they
   shift concept subtype, price tier, or risk tolerance.

A map is still useful, but it supports the shortlist instead of being the
primary interface. Motivating example: a campus-adjacent zone like NYU Tandon
/ MetroTech where dense lunch demand coexists with a weak healthy-food mix.

## 4. Core ML Stack

### 4.1 Neighborhood Phase Discovery

- Build a neighborhood-year panel from permits, licenses, inspections, ACS,
  PLUTO, mobility, and housing-pressure features.
- Use k-means and Gaussian Mixture Models to discover neighborhood regimes
  (k=3 and k=4 evaluated).
- Validate regime assignments against NYU Furman Center neighborhood
  narratives.

### 4.2 Restaurant Survival Modeling

- Use official NYC business-license activity as the primary restaurant
  universe.
- Train Cox Proportional Hazards and Random Survival Forest baselines.
- Use neighborhood regime, competition, inspection, and rent-pressure
  features as covariates.
- Current performance: **C-index ≈ 0.80**.

### 4.3 NLP and Demand Signals

- Use Gemini Flash / Flash-Lite to generate silver sentiment labels for review
  text.
- Audit and retain only high-confidence labels.
- Manually label a small gold evaluation set (200–300 reviews).
- Aggregate labels directly into healthy-demand features instead of training
  a custom transformer in the main plan.
- Use Reddit only as a coarse-geography mention signal (spaCy NER + CD-level
  aggregation), with **NYC 311 as the documented fallback**.

### 4.4 Final Ranking Layer

- MVP: interpretable healthy-food white-space score.
- Stretch: LambdaMART learning-to-rank head — **shipped** as
  `data/models/ranking_model.joblib`.
- Practical recommendation unit: a **micro-zone** (10-minute walk shed, lunch
  corridor, business-district catchment, or small grid cell).

The score is **subtype-aware**: a zone can have several popular healthy bowl
concepts and still be underserved for a different healthy subtype such as
healthy Indian fast casual.

Detailed model contracts: [`docs/ModelInterfaces.md`](docs/ModelInterfaces.md).

## 5. Data Strategy

**Primary sources** (all integrated):

- NYC DOB permits
- NYC DCWP/DCA Legally Operating Businesses
- NYC DOHMH restaurant inspection results
- U.S. Census ACS 5-year estimates
- NYC PLUTO / MapPLUTO
- Inside Airbnb (subject to historical coverage)
- Citi Bike trip / station data
- NYC 311 complaints

**Conditional enrichment**:

- Yelp Fusion API + Yelp Open Dataset (only after NYC coverage audit)
- Reddit neighborhood mentions at Community District level

**Removed**: Google Trends.

Locked temporal window: **2020–2024** (set in `src/config/constants.py`).
No random splits — blocked / rolling temporal backtests only. Per-source
audit details: [`docs/temporal_audit.md`](docs/temporal_audit.md).

Authoritative column-by-column schema: [`docs/DataDictionary.md`](docs/DataDictionary.md).

---

## 6. Repository Structure

The repository follows a strict separation between **data**, **library code
under `src/`**, **product surfaces (`frontend/`, `src/api/`)**, **operational
scripts**, **tests**, and **docs**. Nothing in `src/` does network I/O at
import time, and `frontend/` only consumes the FastAPI surface — it never
reaches into `src/data/` directly.

```text
CS473-FML/
├── README.md                    # This file: overview, setup, structure, labor, sprints
├── Makefile                     # Entry points: make etl / train / api / ui / test
├── requirements.txt             # Pinned Python deps (Python 3.11+)
├── pytest.ini / .coveragerc     # Test runner + coverage config
├── ruff.toml                    # Lint / format config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .env.example                 # Template for secrets (e.g. GEMINI_API_KEY)
├── run_full_pipeline.py         # End-to-end driver: ETL → feature matrix → training
│
├── data/                        # Persisted artifacts (gitignored except small fixtures)
│   ├── raw/                     # Date-stamped immutable source extracts
│   ├── processed/               # Canonical parquet tables (feature_matrix, ground truth)
│   ├── geojson/                 # NTA + Community District boundaries
│   └── models/                  # Trained joblib artifacts
│
├── docs/                        # Design, proposal, dictionaries, evaluation reports
│
├── frontend/                    # Streamlit shortlist-first UI (consumes the FastAPI backend only)
│   ├── app.py                   # App entry point
│   ├── components/              # Reusable widgets (input form, recommendation cards, map, scenario panel)
│   ├── pages/                   # Multi-page Streamlit pages
│   ├── views/                   # Long-form static content
│   └── utils/                   # Frontend-only helpers
│
├── notebooks/                   # Exploratory analyses (read-only narrative)
│
├── scripts/                     # One-off CLIs (run_api.sh, smoke_api.py, data fixers)
│
├── src/                         # All importable library code
│   ├── api/                     # FastAPI service: contract layer between models and frontend
│   │   └── routers/             # /predict/cmf, /predict/trajectory, /shortlist, /scenarios, /health
│   ├── config/                  # constants.py, settings.py
│   ├── data/                    # ETL: one module per source + quality + audit + registry
│   ├── features/                # Feature engineering on top of cleaned ETL outputs
│   ├── models/                  # Trajectory, survival, scoring, ranking, explainability
│   ├── nlp/                     # Gemini labels, review aggregates, embeddings, NER
│   ├── pipeline/                # Cross-cutting orchestration helpers
│   ├── schemas/                 # Pydantic request / response / dataset schemas
│   ├── utils/                   # Geospatial, taxonomy, paths
│   └── validation/              # Backtesting, ablation, causal, evaluation CLIs
│
└── tests/                       # 606 pytest cases mirroring src/ layout one-to-one
```

Per-file annotations live in [`docs/Design.md`](docs/Design.md) §1.

### Separation-of-concerns guarantees

- `src/data/` is the only place that talks to external sources.
- `src/features/` is pure pandas/numpy on top of `data/processed/`; no
  re-fetching.
- `src/models/` consumes the feature matrix and writes joblib artifacts to
  `data/models/`; no network, no UI.
- `src/api/` is the only public network surface.
- `frontend/` only calls the API.
- `tests/` mirrors `src/` so every module has an obvious test home.

---

## 7. Division of Labor

Every team member owns at least one concrete module path **and** at least one
shippable deliverable. Source of truth: [`docs/Design.md`](docs/Design.md) §2.

| Member | Role | Primary modules | Deliverables |
| :---- | :---- | :---- | :---- |
| **Harsh Agarwal** | Backend / ML — Survival & Scoring | `src/models/survival_model.py`, `train_survival.py`, `cmf_score.py`, `train_scoring.py`, `ranking_model.py`, `explainability.py` | `data/models/{survival,scoring,ranking}_model.joblib`; survival C-index ≈ 0.80 in `docs/EvaluationResults.md` |
| **Siqi Zhu** | Backend / ML — Phase Discovery & Validation | `src/models/trajectory_model.py`, `src/validation/{backtesting,ablation,causal,run_evaluation}.py` | k-means + GMM clusters; `notebooks/02_trajectory_model.ipynb`; backtest / ablation parquets; `docs/CausalMLEvaluationReport.md` |
| **Tony Zhao** | Data / ETL Lead | `src/data/etl_*.py` (all 10 sources), `etl_runner.py`, `quality.py`, `audit.py`, `scripts/*` | All ETL parquets in `data/processed/`; `docs/temporal_audit.md`; source sections of `docs/DataDictionary.md` |
| **Amanda Dong** | Frontend / NLP | `frontend/app.py`, `frontend/components/*`, `frontend/pages/*`, `src/nlp/{gemini_labels,review_aggregates,subtype_classifier,neighborhood_mentions}.py` | Streamlit UI; `data/processed/gemini_full_zone_features.csv`; `notebooks/04_nlp.ipynb` |
| **Catherine Yi** | Project Lead / Integration | `src/api/main.py`, `src/api/routers/recommendations.py`, `src/features/{feature_matrix,ground_truth,zone_crosswalk}.py`, `run_full_pipeline.py`, `docs/*` | API contract (`docs/api_contract.md`); canonical `feature_matrix.parquet` (726 × 49); `run_full_pipeline.py`; `docs/ReportSections.md`, `docs/Presentation.md` |

Each member also owns the matching `tests/test_*.py` for their primary
modules. Total tests: **606 passing**.

---

## 8. Sprint Plan & Status

Spring 2026 · 8 weeks compressed into 4 sprint blocks. Full per-sprint detail
and completion notes: [`docs/Sprints.md`](docs/Sprints.md).

| Sprint | Theme | Status |
| :---- | :---- | :---- |
| **Sprint 1** | Source audit, setup, feasibility lock | ✅ Shipped — taxonomy, `uv` workflow, ETL pilots, locked 2020–2024 window |
| **Sprint 2** | Feature matrix + neighborhood phase discovery | ✅ Shipped — 726 × 49 zone-year matrix, k-means / GMM clusters, 137 micro-zones, Google Trends removed, 311 fallback wired |
| **Sprint 3** | Survival modeling, NLP, product integration | ✅ Shipped — Cox PH + RSF (C-index ≈ 0.80), Gemini labels on full Yelp corpus, FastAPI endpoints, Streamlit UI |
| **Sprint 4** | Backtesting, robustness, final packaging | ✅ Shipped — temporal backtests, ablations, causal checks, shortlist-first UI, report + slides |

### Non-negotiable rules (still enforced)

- No Google Trends.
- No Reddit as a core signal until sparsity audit is complete.
- No Yelp-as-universe until NYC coverage audit is done.
- No random train/test split for the headline result.
- Weak-history sources are downgraded to static covariates or replaced with
  the documented fallback.
- Product framing is healthy-food white space, not a generic recommender.

---

## 9. Setup

Use **Python 3.11+** (several pinned packages do not support 3.9). You can
use either `uv` or a conda/venv + `pip` workflow; pick one and stay
consistent.

**Option A — `uv` (team default)**

1. Install `uv`.
2. `uv venv`
3. Activate: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows).
4. `uv pip install -r requirements.txt`
5. Tests: `uv run pytest`
6. **Streamlit UI** (frontend): from the repo root, start the backend in one
   terminal (`uv run python -m uvicorn src.api.main:app --reload --port 8000`),
   then in another terminal run
   `uv run python -m streamlit run frontend/app.py` — the app opens in the
   browser (default http://localhost:8501). Same as `make ui` after `make api`.

**Option B — conda + pip**

1. `conda create -n cs473-fml python=3.11 -y && conda activate cs473-fml`
2. `python -m pip install -U pip setuptools wheel`
3. `python -m pip install -r requirements.txt`
4. Tests: `python -m pytest`
5. **Streamlit UI** (frontend): start the API in one terminal
   (`python -m uvicorn src.api.main:app --reload --port 8000`), then in another
   run `python -m streamlit run frontend/app.py` (browser at
   http://localhost:8501 by default).

If `pip` cannot resolve a pin in `requirements.txt`, treat it as a **team
coordination** item — do not silently change shared pins on your own branch.

A `GEMINI_API_KEY` is required for live NLP labeling
(`src/nlp/gemini_labels.py`). All other features run without API keys. Copy
`.env.example` → `.env` and fill in.

## 10. Quick Start

The `Makefile` is the single CLI surface:

| Command | What it does |
| :---- | :---- |
| `make install` | `uv pip install -r requirements.txt` |
| `make etl` | Run all ETL modules and write `data/processed/*.parquet` |
| `make etl-small` | ETL with `--limit 5000` (fast smoke run) |
| `make pipeline` | Full ETL → feature matrix → training |
| `make train` | Train from existing parquets only |
| `make api` | Boot FastAPI on port 8000 |
| `make ui` | Boot Streamlit frontend |
| `make run` | Start API + UI together |
| `make test` | Run the 606-case pytest suite |
| `make coverage` | Tests with coverage report |
| `make lint` / `make format` | `ruff` check / format |

API contract: [`docs/api_contract.md`](docs/api_contract.md).
Smoke check: `python scripts/smoke_api.py`.

---

## 11. Repository Status & Artifacts

The implementation is complete across all eight planned stages:

1. **Data source audit** — 10 ETL modules with real NYC Open Data
   integrations.
2. **Canonical neighborhood feature matrix** — 726 zone-year rows, 49
   features at `data/processed/feature_matrix.parquet`.
3. **Micro-zone layer** — 137 zones across campus, lunch-corridor,
   transit-catchment, and business-district types.
4. **Phase discovery** — k-means and GMM trajectory clustering; NTA
   healthy-food cluster assignments.
5. **Survival modeling** — Cox PH + Random Survival Forest;
   `data/models/survival_model.joblib` (C-index ≈ 0.80).
6. **NLP labeling and aggregation** — Gemini Flash silver labels on full Yelp
   corpus; zone-level rollups in
   `data/processed/gemini_full_zone_features.csv`.
7. **Healthy-food white-space ranking** — XGBoost scoring
   (`scoring_model.joblib`) + LambdaMART ranker (`ranking_model.joblib`);
   interpretable CMF score.
8. **API and Streamlit integration** — FastAPI backend, shortlist-first
   Streamlit UI.

**Evaluation artifacts**: `data/processed/backtest_results.parquet`,
`ablation_results.parquet`, [`docs/EvaluationResults.md`](docs/EvaluationResults.md),
[`docs/CausalMLEvaluationReport.md`](docs/CausalMLEvaluationReport.md).

**Test suite**: 606 tests passing.

---

## 12. Documentation Index

| Document | Purpose |
| :---- | :---- |
| [`docs/Proposal.md`](docs/Proposal.md) | Problem framing, methods, research-driven choices |
| [`docs/Design.md`](docs/Design.md) | Repository structure, division of labor, environment and readiness |
| [`docs/Sprints.md`](docs/Sprints.md) | Sprint-by-sprint plan with completion status |
| [`docs/Research.md`](docs/Research.md) | Research, data & modeling rationale |
| [`docs/DataDictionary.md`](docs/DataDictionary.md) | Authoritative column-by-column schema |
| [`docs/ModelInterfaces.md`](docs/ModelInterfaces.md) | Exact model I/O contracts and runtime behavior |
| [`docs/api_contract.md`](docs/api_contract.md) | FastAPI endpoint shapes |
| [`docs/EvaluationResults.md`](docs/EvaluationResults.md) | Backtest, ablation, ranking metrics |
| [`docs/CausalMLEvaluationReport.md`](docs/CausalMLEvaluationReport.md) | Causal-validation findings |
| [`docs/temporal_audit.md`](docs/temporal_audit.md) | Per-source coverage + cutoff decisions |
| [`docs/ReportSections.md`](docs/ReportSections.md) | Final-report draft material |
| [`docs/Presentation.md`](docs/Presentation.md) | Slide deck outline |
