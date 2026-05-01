# NYC Restaurant Intelligence Platform — Design Document

Updated: April 30, 2026
Team: Catherine · Harsh · Tony · Siqi · Amanda
Repo: <https://github.com/Amanda-dong/CS473-FML>

Design reference for collaborators and reviewers. It covers (1) repository
structure with a one-line description of each component, (2) division of labor
across the five-person team with concrete module ownership, and (3) how the repo
documents setup and maps documentation to runnable code (README,
`requirements.txt`, environment, modules under `src/`).

Longer engineering details live in `docs/DataDictionary.md`,
`docs/ModelInterfaces.md`, and `docs/api_contract.md`. Architecture rationale
sits in `docs/Research.md` and `docs/Proposal.md`.

---

## 1. Repository Structure

The repository follows a strict separation between **data**, **library code
under `src/`**, **product surfaces (`frontend/`, `src/api/`)**, **operational
scripts**, **tests**, and **docs**. Nothing in `src/` writes to the network at
import time, and nothing in `frontend/` reaches into `src/data/` directly — the
API layer is the contract between them.

```text
CS473-FML/
├── README.md                    # Project overview, setup, quick start, doc index
├── Makefile                     # One-line entry points: `make etl`, `make train`, `make api`, `make ui`, `make test`
├── requirements.txt             # Pinned Python deps (Python 3.11+)
├── pytest.ini / .coveragerc     # Test runner + coverage config
├── ruff.toml                    # Lint/format config (single source of truth for style)
├── .pre-commit-config.yaml      # Pre-commit hooks (ruff, formatting, commitlint)
├── .env.example                 # Template for secrets (e.g. GEMINI_API_KEY)
├── run_full_pipeline.py         # End-to-end orchestrator: ETL → feature matrix → training
│
├── data/                        # All persisted artifacts (gitignored except small fixtures)
│   ├── raw/                     # Date-stamped immutable source extracts (CSV/JSON from NYC Open Data, Yelp, ACS)
│   ├── processed/               # Canonical parquet tables (feature_matrix.parquet, ground truth, ETL outputs)
│   ├── geojson/                 # NTA + Community District boundaries used as join infrastructure
│   └── models/                  # Trained joblib artifacts: scoring_model, survival_model, ranking_model
│
├── docs/                        # Design, proposal, data dictionary, model interfaces, evaluation reports
│   ├── Proposal.md              # Problem framing, methods, research-driven choices
│   ├── Design.md                # This file: structure, labor, repo readiness
│   ├── Sprints.md               # Sprint-by-sprint task division and completion status
│   ├── Research.md              # Research, data viability, dependency notes
│   ├── DataDictionary.md        # Authoritative column-by-column schema reference
│   ├── ModelInterfaces.md       # Exact model I/O contracts and runtime behavior
│   ├── EvaluationResults.md     # Backtest, ablation, and ranking metrics
│   ├── CausalMLEvaluationReport.md # Causal-validation findings
│   ├── api_contract.md          # FastAPI endpoint shapes
│   ├── ReportSections.md        # Final report draft material
│   ├── Presentation.md          # Slide deck outline
│   └── temporal_audit.md        # Per-source coverage + cutoff decisions
│
├── frontend/                    # Streamlit shortlist-first UI (consumes the FastAPI backend only)
│   ├── app.py                   # Entry point: routes, state, calls /predict/cmf and /predict/trajectory
│   ├── components/              # Reusable widgets: input_form, recommendation_card, scenario_panel, map_view, results_panel, data_freshness
│   ├── pages/                   # Multi-page Streamlit pages (e.g. Methodology)
│   ├── views/                   # Static long-form content (e.g. methodology copy)
│   └── utils/                   # Frontend-only helpers (search state)
│
├── notebooks/                   # Exploratory analysis (read-only narratives, not part of runtime path)
│   ├── 01_eda.ipynb             # Data audit + coverage exploration
│   ├── 02_trajectory_model.ipynb # k-means / GMM phase discovery
│   ├── 03_survival_model.ipynb  # Cox PH + RSF prototyping
│   ├── 04_nlp.ipynb             # Gemini labeling + aggregation
│   └── 05_backtesting.ipynb     # Temporal validation
│
├── scripts/                     # One-off CLIs and operational helpers
│   ├── run_api.sh               # Boots uvicorn against `src.api.main:app`
│   ├── smoke_api.py             # Hits each API endpoint as a smoke test
│   ├── filter_yelp_reviews_fusion.py # Pre-filter Yelp corpus to NYC restaurants
│   ├── join_reviews_to_zones.py # Spatially join reviews onto micro-zones
│   ├── assign_yelp_business_zones.py # Map Yelp businesses to zone_id
│   ├── download_nta_geojson.py  # Fetch boundary GeoJSON
│   ├── fill_inspections_data.py # Backfill inspection rollups
│   ├── fix_nulls.py             # Re-run null repair on existing parquets
│   └── verify_diversity.py      # Audit recommendation diversity
│
├── src/                         # All importable library code
│   ├── api/                     # FastAPI service: contract layer between models and frontend
│   │   ├── main.py              # App factory, middleware, router registration
│   │   ├── deps.py              # Shared dependency injection (model loaders, settings)
│   │   └── routers/
│   │       ├── recommendations.py # /predict/cmf, /predict/trajectory, /shortlist, /scenarios
│   │       ├── datasets.py       # Dataset metadata + freshness endpoints
│   │       └── health.py         # Liveness/readiness probes
│   │
│   ├── config/                  # Static project configuration
│   │   ├── constants.py         # Temporal window, scoring weights, taxonomy thresholds
│   │   └── settings.py          # Pydantic settings reading .env
│   │
│   ├── data/                    # ETL: one module per source, plus quality + audit
│   │   ├── etl_runner.py        # Orchestrates all ETL modules with consistent logging
│   │   ├── etl_licenses.py      # NYC DCWP Legally Operating Businesses
│   │   ├── etl_permits.py       # NYC DOB building permits
│   │   ├── etl_inspections.py   # NYC DOHMH restaurant inspections
│   │   ├── etl_pluto.py         # MapPLUTO lot-level land-use
│   │   ├── etl_acs.py           # Census ACS 5-year demographics
│   │   ├── etl_yelp.py          # Yelp Open Dataset (audited NYC slice)
│   │   ├── etl_citibike.py      # Citi Bike trip + station mobility
│   │   ├── etl_airbnb.py        # Inside Airbnb housing pressure proxy
│   │   ├── etl_311.py           # NYC 311 complaints (Reddit fallback)
│   │   ├── etl_boundaries.py    # NTA + CD boundary geometry
│   │   ├── audit.py             # Per-source coverage + freshness diagnostics
│   │   ├── quality.py           # Null-fill, schema validation, dtype coercion
│   │   ├── registry.py          # Source ↔ module mapping for `etl_runner`
│   │   └── enrich_zone_features.py # Post-ETL zone-level enrichment
│   │
│   ├── features/                # Feature engineering on top of cleaned ETL outputs
│   │   ├── feature_matrix.py    # Builds the canonical zone-year matrix (49 cols × 726 rows)
│   │   ├── ground_truth.py      # y_composite construction + label_quality
│   │   ├── microzones.py        # Walk-shed and corridor definitions
│   │   ├── zone_crosswalk.py    # NTA ↔ zone_id mapping infrastructure
│   │   ├── healthy_gap.py       # Healthy-supply-gap feature
│   │   ├── competition_score.py # Local competitive intensity
│   │   ├── demand_signals.py    # Review-derived demand aggregates
│   │   ├── license_velocity.py  # Open/close rate features
│   │   ├── rent_trajectory.py   # Rent / assessed-value pressure proxies
│   │   ├── merchant_viability.py # Composite merchant-side risk
│   │   └── yelp_microzones.py   # Yelp-business → micro-zone roll-ups
│   │
│   ├── models/                  # Modeling layer + training entry points
│   │   ├── trajectory_model.py  # k-means / GMM phase discovery
│   │   ├── survival_model.py    # Cox PH + Random Survival Forest
│   │   ├── train_survival.py    # Survival training CLI
│   │   ├── cmf_score.py         # Interpretable concept-market-fit score
│   │   ├── train_scoring.py     # XGBoost scoring training CLI
│   │   ├── ranking_model.py     # LambdaMART learning-to-rank head
│   │   ├── explainability.py    # SHAP-style driver attribution
│   │   ├── model_loader.py      # Lazy joblib loader shared by API + scripts
│   │   └── baselines.py         # Naïve/sanity baselines
│   │
│   ├── nlp/                     # Text-side pipeline
│   │   ├── gemini_labels.py     # Gemini Flash silver-label generator
│   │   ├── review_aggregates.py # Zone-level healthy-demand rollups
│   │   ├── embeddings.py        # Lightweight CPU embeddings
│   │   ├── topic_model.py       # Optional topic-cluster summary
│   │   ├── subtype_classifier.py # Healthy subtype assignment
│   │   ├── neighborhood_mentions.py # spaCy NER for Reddit / 311 mentions
│   │   ├── sentiment.py         # Lightweight sentiment helper
│   │   └── white_space.py       # Subtype-gap derivation
│   │
│   ├── pipeline/                # Cross-cutting orchestration helpers
│   │   ├── orchestrator.py      # Stage runner used by `run_full_pipeline`
│   │   ├── preflight.py         # Pre-run environment + data sanity checks
│   │   └── stages.py            # Stage enum / metadata
│   │
│   ├── schemas/                 # Pydantic request/response + dataset schemas
│   │   ├── requests.py          # API request bodies
│   │   ├── results.py           # API response shapes (recommendations, scenarios)
│   │   └── datasets.py          # ETL output schemas
│   │
│   ├── utils/                   # Shared utilities (no upward dependencies)
│   │   ├── geospatial.py        # CRS handling + spatial joins
│   │   ├── taxonomy.py          # Healthy-food subtype taxonomy + matchers
│   │   └── paths.py             # Centralized path constants
│   │
│   └── validation/              # Time-aware evaluation, ablations, causal checks
│       ├── backtesting.py       # Blocked / rolling temporal backtests
│       ├── ablation.py          # Feature-family ablation harness
│       ├── causal.py            # Causal robustness checks
│       ├── run_evaluation.py    # End-to-end evaluation CLI
│       └── run_causal_evaluation.py # Causal evaluation CLI
│
└── tests/                       # 600+ pytest cases (one test_*.py per src subpackage)
    ├── conftest.py              # Shared fixtures (synthetic ETL frames, monkeypatched paths)
    ├── test_etl.py              # ETL module behavior + schema enforcement
    ├── test_features.py         # Feature matrix + ground truth correctness
    ├── test_models.py           # Trajectory, survival, scoring, ranking models
    ├── test_nlp.py              # Gemini labels + aggregates
    ├── test_api.py              # FastAPI endpoint contracts
    ├── test_validation.py       # Backtesting + ablation
    ├── test_causal_validation.py # Causal-evaluation harness
    ├── test_pipeline.py         # End-to-end pipeline integration
    ├── test_geospatial.py       # CRS + spatial-join utilities
    ├── test_zone_crosswalk.py   # NTA ↔ zone_id consistency
    ├── test_enrich_zone_features.py # Zone-level enrichment
    └── test_frontend_search_state.py # Streamlit search-state helper
```

### Separation-of-concerns guarantees

- **`src/data/`** is the only place that talks to external sources. Nothing
  downstream of it does I/O against URLs.
- **`src/features/`** is pure pandas/numpy on top of `data/processed/`. It never
  re-fetches.
- **`src/models/`** consumes the feature matrix and writes joblib artifacts to
  `data/models/`; it never fetches and never renders UI.
- **`src/api/`** is the only public network surface. It loads models lazily via
  `src/models/model_loader.py`.
- **`frontend/`** only calls the API. It does not import from `src/data/`,
  `src/features/`, or `src/models/`.
- **`tests/`** mirrors the `src/` package layout one-to-one, so every module has
  a clear test home.

---

## 2. Division of Labor

Every team member owns at least one concrete module path **and** at least one
shippable deliverable. The `Primary modules` column lists the directories or
files where that member leads review; the `Deliverables` column lists the main
artifacts that illustrate that ownership.

| Member | Role | Primary modules (paths) | Deliverables |
| :---- | :---- | :---- | :---- |
| **Harsh Agarwal** | Backend / ML — Survival & Scoring | `src/models/survival_model.py`, `src/models/train_survival.py`, `src/models/cmf_score.py`, `src/models/train_scoring.py`, `src/models/ranking_model.py`, `src/models/explainability.py` | `data/models/survival_model.joblib`, `data/models/scoring_model.joblib`, `data/models/ranking_model.joblib`; survival C-index ≈ 0.80 reported in `docs/EvaluationResults.md` |
| **Siqi Zhu** | Backend / ML — Phase Discovery & Validation | `src/models/trajectory_model.py`, `src/validation/backtesting.py`, `src/validation/ablation.py`, `src/validation/causal.py`, `src/validation/run_evaluation.py` | k-means + GMM trajectory clusters in `notebooks/02_trajectory_model.ipynb`; backtest + ablation parquets under `data/processed/`; `docs/CausalMLEvaluationReport.md` |
| **Tony Zhao** | Data / ETL Lead | `src/data/etl_*.py` (licenses, permits, inspections, pluto, acs, citibike, airbnb, 311, boundaries), `src/data/etl_runner.py`, `src/data/quality.py`, `src/data/audit.py`, `scripts/*` | All ETL parquets in `data/processed/`; `docs/temporal_audit.md`; `docs/DataDictionary.md` source sections |
| **Amanda Dong** | Frontend / NLP | `frontend/app.py`, `frontend/components/*`, `frontend/pages/*`, `src/nlp/gemini_labels.py`, `src/nlp/review_aggregates.py`, `src/nlp/subtype_classifier.py`, `src/nlp/neighborhood_mentions.py` | Streamlit shortlist-first UI; `data/processed/gemini_full_zone_features.csv`; `notebooks/04_nlp.ipynb` |
| **Catherine Yi** | Project Lead / Integration | `src/api/main.py`, `src/api/routers/recommendations.py`, `src/features/feature_matrix.py`, `src/features/ground_truth.py`, `src/features/zone_crosswalk.py`, `run_full_pipeline.py`, `docs/*` | API contract (`docs/api_contract.md`); canonical `feature_matrix.parquet` (726 × 49); end-to-end `run_full_pipeline.py`; final `docs/ReportSections.md` and `docs/Presentation.md` |

Cross-cutting ownership: every member is responsible for the `tests/test_*.py`
file that mirrors their primary module(s). Test count today: **606 passing**.

---

## 3. Repository and Environment Readiness

Public repo: <https://github.com/Amanda-dong/CS473-FML>.

| Item | Status | Notes |
| :---- | :---- | :---- |
| README | ✅ | Overview, setup (uv + conda paths), quick start, structure summary, documentation index |
| Dependencies | ✅ | `requirements.txt` (Python 3.11+, pinned deps); `Makefile` provides `make install` |
| Reproducible environment | ✅ | `uv venv` + `.env.example`; `make test` runs the pytest suite (606 passing); `make api` and `make ui` start the prototype |
| Code layout aligned with §1 | ✅ | Each described area maps to runnable modules under `src/` (`__init__.py` where needed); `tests/` mirrors the package layout |

The codebase is actively implemented—not a skeleton. Concretely:

- **ETL**: 10 source modules under `src/data/`, all callable through
  `etl_runner.py`. Real fetch paths for `permits`, `licenses`, `inspections`,
  `pluto`, `acs`, `yelp`, `citibike`, `airbnb`, `311`, `boundaries`.
- **Features**: `feature_matrix.py` produces a 726-row × 49-column zone-year
  matrix saved to `data/processed/feature_matrix.parquet`. Ground truth and
  micro-zone crosswalks are implemented.
- **Models**: trajectory clustering (k-means + GMM), survival (Cox PH + RSF),
  scoring (XGBoost), ranking (LambdaMART), and explainability are all wired up.
  Trained artifacts ship in `data/models/`.
- **NLP**: full Yelp-corpus Gemini labeling pipeline; zone-level rollups in
  `data/processed/gemini_full_zone_features.csv`.
- **API + UI**: FastAPI service in `src/api/` (`/predict/cmf`,
  `/predict/trajectory`, `/shortlist`, `/scenarios`) consumed by the Streamlit
  app in `frontend/`.
- **Validation**: blocked temporal backtesting, feature-family ablations, and a
  causal-robustness harness under `src/validation/`.

---

## 4. Architecture Decisions (reference)

This section captures the rationale for the non-obvious choices baked into the
structure above. It is intentionally short — the long version lives in
`docs/Research.md` and `docs/Proposal.md`.

| Area | Decision | Why |
| :---- | :---- | :---- |
| Phase labels | Unsupervised k-means + GMM, not hand-labeled supervised classes | Removes the weakest assumption in the original proposal; lets the data define regimes before interpretation |
| Survival backbone | NYC DCWP licensing as the primary restaurant universe | Better coverage and temporal completeness than Yelp alone |
| Sentiment labels | Gemini silver labels + small gold audit, no transformer fine-tuning in main plan | Preserves NLP signal while keeping compute CPU-friendly |
| Reddit signal | spaCy NER + Community District aggregation; binary recent-mention feature; 311 fallback | More tractable than geocoding sparse social text to 195 NTAs |
| Google Trends | Removed | `pytrends` is unofficial; neighborhood-level signal quality is weak |
| Recommendation unit | Micro-zones (walk sheds, corridors, campus catchments) — not whole NTAs | Merchants choose between walkable lunch catchments, not borough-sized areas |
| Validation | Blocked / rolling temporal backtests, no random splits | Prevents leakage across years and matches deployment reality |

---

## 5. Evaluation Targets

- **Clustering**: stability + interpretability against NYU Furman Center
  narratives (not silhouette alone).
- **Survival**: concordance index + calibration; current C-index ≈ 0.80.
- **NLP**: agreement between Gemini silver labels and a 200–300-row gold set;
  stability of zone-level aggregates.
- **Final ranking**: top-k usefulness (NDCG@k / recall@k) on held-out periods
  when the learned ranker is enabled.
- **Product**: case-study sanity checks on motivating zones (NYU Tandon /
  MetroTech, FiDi lunch corridor, etc.).

Detailed numbers live in `docs/EvaluationResults.md` and
`docs/CausalMLEvaluationReport.md`.

---

## 6. Engineering Standards

- Python 3.11+; `uv` is the team-default environment manager; `pip + venv` is
  supported as a fallback.
- Raw data is immutable and date-stamped; never edited in place.
- No random train/test splits in headline evaluation — temporal blocks only.
- Lint and format with `ruff` (config in `ruff.toml`); pre-commit hooks
  enforce both before push.
- 600+ pytest cases gate every PR; coverage tracked via `.coveragerc`.
- API keys (Gemini, Yelp Fusion) read from `.env` via
  `src/config/settings.py`; never committed.
