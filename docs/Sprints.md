# NYC Restaurant Intelligence Platform — Sprint Plan

Updated: April 30, 2026
Spring 2026 · 8 weeks compressed into 4 sprint blocks
Team: Catherine · Harsh · Tony · Siqi · Amanda

This document is the historical sprint plan **plus** a per-sprint completion
status report (✅ shipped, ⚠️ partial, ❌ cut). For static module ownership see
`docs/Design.md` §2; for product/methodology rationale see `docs/Proposal.md`.

---

## Sprint 1 — Source Audit, Setup, and Feasibility Lock ✅

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Define the neighborhood-year schema and the healthy-food scoring logic | Run source-by-source ETL feasibility and coverage checks | Own the audit matrix, data dictionary, and final go / no-go decisions |
| Required work | Prototype k-means and GMM on a small ACS + PLUTO + permits sample; define candidate feature families; define a healthy-food taxonomy and white-space formula; add subtype rules for Mediterranean bowls vs. healthy Indian / South Asian bowls; write criteria for selecting cluster count | Set up the repo workflow with `uv`; pull pilot slices for permits, licenses, inspections, Citi Bike, Inside Airbnb, and Yelp; audit Yelp Open Dataset NYC coverage immediately; define candidate micro-zones around campuses and lunch corridors; test Reddit collection only as a pilot | Run the one-day temporal audit sprint; record earliest year, cadence, spatial unit, and fallback per source; define NTA / CD crosswalk assets; pick motivating case-study zones |
| Deliverables | Draft phase-discovery notebook, healthy-food taxonomy, subtype rules, and feature spec | Data coverage memo, pilot ETL outputs, and candidate micro-zone layer | Approved source inventory and locked temporal evaluation window |

**Completion status:**

- ✅ Healthy-food taxonomy shipped in `src/utils/taxonomy.py`.
- ✅ k-means / GMM prototype lives in `notebooks/02_trajectory_model.ipynb`
  and `src/models/trajectory_model.py`.
- ✅ `uv`-based repo workflow + `Makefile` entry points (`make etl`,
  `make train`, `make api`, `make ui`, `make test`).
- ✅ ETL pilot slices for all 10 planned sources; module map in
  `src/data/etl_*.py`.
- ✅ Temporal audit: `docs/temporal_audit.md`. Locked window: 2020–2024
  (set in `src/config/constants.py`).
- ✅ Crosswalk implementation: `src/features/zone_crosswalk.py` (NTA ↔
  zone_id) plus `data/geojson/`.

---

## Sprint 2 — Feature Matrix and Neighborhood Phase Discovery ✅

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Build the first full neighborhood panel and discover neighborhood regimes | Finish production-ready ETL and the healthy-food supply-gap layer | Review joins, labeling logic, and validation criteria |
| Required work | Engineer permit velocity, license dynamics, inspection aggregates, rent proxies, and mobility features; run k-means and GMM; compare cluster stability; assign post-hoc labels | Add Airbnb density where coverage supports it; remove Google Trends entirely; build spaCy NER + neighborhood lookup for Reddit; aggregate Reddit at CD level as a binary recent-mention signal; prepare 311 fallback ETL; build micro-zone walk sheds; classify nearby restaurants into healthy categories and subtypes; validate spatial joins | Spot-check clusters against NYU Furman Center reports; document why each cluster label is defensible; approve the main feature matrix, micro-zone schema, and imputation plan |
| Deliverables | Clustered neighborhood panel with interpreted regimes | Validated ETL stack, healthy-food supply-gap features, subtype-gap features, and micro-zone layer | Cluster-validation memo and feature governance notes |

**Completion status:**

- ✅ Feature engineering modules: `src/features/license_velocity.py`,
  `rent_trajectory.py`, `competition_score.py`, `healthy_gap.py`,
  `merchant_viability.py`, `demand_signals.py`.
- ✅ Canonical zone-year matrix: 726 rows × 49 columns at
  `data/processed/feature_matrix.parquet`.
- ✅ Phase discovery shipped (k=3 and k=4 evaluated). Trajectory clusters
  surfaced via `src/models/trajectory_model.py`.
- ✅ Google Trends removed; not present anywhere in `src/`.
- ✅ Reddit NER (`src/nlp/neighborhood_mentions.py`) + 311 fallback
  (`src/data/etl_311.py`).
- ✅ 137 micro-zones in `src/features/microzones.py` + `yelp_microzones.py`.
- ✅ Imputation governance: `src/data/quality.py::fill_feature_matrix_nulls`.

---

## Sprint 3 — Survival Modeling, NLP, and Product Integration ✅

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Build restaurant survival baselines and combine them with healthy-food white-space logic | Build the realistic NLP pipeline and connect the UI to live outputs | Keep the end-to-end pipeline coherent and report-ready |
| Required work | Use official NYC licensing data as the primary restaurant universe; fit Cox PH and RSF baselines; combine regime features with competition and rent burden; define the first healthy-food opening score with subtype-level gaps; if time allows, prototype a small CPU-friendly ranking layer; expose `/predict/trajectory` and `/predict/cmf` endpoints | Use Gemini Flash / Flash-Lite to generate silver labels for Yelp reviews; retain only high-confidence examples after audit; manually annotate 200–300 gold examples for held-out evaluation; aggregate labels into healthy-demand features; build white-space and competitor features; connect frontend components to backend responses | Review survival target construction; sign off on the healthy-food opening-score assumptions; run end-to-end QA across ETL, model, API, and Streamlit layers; maintain the bug tracker |
| Deliverables | Survival models, healthy-food score prototype, subtype-aware recommendations, and backend endpoints | NLP models, demand features, and frontend integration | Approved end-to-end prototype and issue log |

**Completion status:**

- ✅ Survival models: `src/models/survival_model.py` + `train_survival.py`;
  artifact `data/models/survival_model.joblib`; **C-index ≈ 0.80**.
- ✅ Scoring model: `src/models/cmf_score.py` + `train_scoring.py`;
  artifact `data/models/scoring_model.joblib`.
- ✅ Ranking model (LambdaMART): `src/models/ranking_model.py`; artifact
  `data/models/ranking_model.joblib`.
- ✅ Gemini labels on full Yelp corpus: `src/nlp/gemini_labels.py`,
  zone-level rollups in `data/processed/gemini_full_zone_features.csv`.
- ✅ FastAPI service in `src/api/`: `/predict/cmf`, `/predict/trajectory`,
  `/shortlist`, `/scenarios`. Contract documented in `docs/api_contract.md`.
- ✅ Streamlit shortlist-first UI: `frontend/app.py` +
  `frontend/components/`. Methodology page under `frontend/pages/`.
- ✅ Explainability: `src/models/explainability.py`.

---

## Sprint 4 — Backtesting, Robustness, and Final Packaging ✅

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Prove the modeling story with time-aware evaluation | Make the demo stable and interpretable | Finalize the report and presentation package |
| Required work | Run blocked / rolling temporal backtests using the audited cutoff; perform ablations on major feature families; generate figures for survival, clustering, and ranking; document residual risks; report ranking metrics | Fix integration bugs; add a data-freshness note to the UI; make the frontend shortlist-first rather than map-first; build evidence cards with healthy supply-gap summaries, risk flags, and confidence; write the data pipeline and NLP sections of the report | Compile all report sections; write the executive summary and conclusion; ensure citations, figures, and tables are consistent; lead rehearsals and lock the final presentation flow |
| Deliverables | Final evaluation package and model artifacts | Stable demo and polished UI | Final report draft, presentation deck, and submission checklist |

**Completion status:**

- ✅ Backtesting harness: `src/validation/backtesting.py`; outputs in
  `data/processed/backtest_results.parquet`.
- ✅ Ablation harness: `src/validation/ablation.py`; outputs in
  `data/processed/ablation_results.parquet`.
- ✅ Causal-robustness checks: `src/validation/causal.py` +
  `docs/CausalMLEvaluationReport.md`.
- ✅ Headline metrics in `docs/EvaluationResults.md`.
- ✅ Data-freshness widget in `frontend/components/data_freshness.py`.
- ✅ Shortlist-first UI flow in Streamlit (`frontend/`).
- ✅ Final-report material: `docs/ReportSections.md`,
  `docs/Presentation.md`.
- ✅ 606-test suite passing under `pytest`.

---

## Cross-Sprint Engineering Hygiene

- All ETL outputs pinned to date-stamped immutable raw extracts.
- Lint + format gated by `ruff` and `pre-commit`.
- API keys (Gemini, Yelp Fusion) live only in `.env` (template:
  `.env.example`); never committed.
- `Makefile` is the single CLI surface; `run_full_pipeline.py` is the
  end-to-end driver.

## Non-Negotiable Project Rules (still enforced)

- Do not reintroduce Google Trends.
- Do not commit to Reddit as a core signal until the sparsity audit is
  complete.
- Do not assume Yelp is the restaurant universe until the NYC coverage audit
  is done.
- Do not use a random train/test split for the headline result.
- If historical depth is weak for a dataset, downgrade it to a static
  covariate or replace it with the documented fallback.
- Do not frame the product as a generic restaurant recommender; keep the
  healthy-food white-space use case explicit.
