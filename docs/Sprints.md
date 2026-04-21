# NYC Restaurant Intelligence Platform - Task Division

Updated: April 1, 2026
Spring 2026 · 8 weeks compressed into 4 sprint blocks
Team: Catherine · Harsh · Tony · Siqi · Amanda

## Sprint 1: Source Audit, Setup, and Feasibility Lock

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Define the neighborhood-year schema and the healthy-food scoring logic | Run source-by-source ETL feasibility and coverage checks | Own the audit matrix, data dictionary, and final go or no-go decisions |
| Required work | Prototype k-means and GMM on a small ACS + PLUTO + permits sample; define candidate feature families; define a healthy-food taxonomy and white-space formula; add subtype rules for categories such as Mediterranean bowls versus healthy Indian or South Asian bowls; write criteria for selecting cluster count | Set up the repo workflow with `uv`; pull pilot slices for permits, licenses, inspections, Citi Bike, Inside Airbnb, and Yelp; audit Yelp Open Dataset NYC coverage immediately; define candidate micro-zones around campuses and lunch corridors; test Reddit collection only as a pilot, not a committed core source | Run the one-day temporal audit sprint; record earliest year, cadence, spatial unit, and fallback for each source; define NTA and Community District crosswalk assets; pick motivating case-study zones such as university-adjacent or office lunch districts |
| Deliverables | Draft phase-discovery notebook, healthy-food taxonomy, subtype rules, and feature spec | Data coverage memo, pilot ETL outputs, and candidate micro-zone layer | Approved source inventory and locked temporal evaluation window |

## Sprint 2: Feature Matrix and Neighborhood Phase Discovery

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Build the first full neighborhood panel and discover neighborhood regimes | Finish production-ready ETL and the healthy-food supply-gap layer | Review joins, labeling logic, and validation criteria |
| Required work | Engineer permit velocity, license dynamics, inspection aggregates, rent proxies, and mobility features; run k-means and GMM; compare cluster stability; assign post-hoc labels to candidate clusters | Add Airbnb density where historical coverage supports it; remove Google Trends entirely; build spaCy NER plus neighborhood lookup for Reddit; aggregate Reddit at Community District level as a binary recent-mention signal; prepare NYC 311 fallback ETL if Reddit is too sparse; build micro-zone walk sheds around campuses and lunch corridors; classify nearby restaurants into healthy categories and subtypes instead of only healthy versus non-healthy; validate all spatial joins | Spot-check clusters against NYU Furman Center reports; document why each cluster label is defensible; approve the main feature matrix, micro-zone schema, and imputation plan |
| Deliverables | Clustered neighborhood panel with interpreted regimes | Validated ETL stack, healthy-food supply-gap features, subtype-gap features, and micro-zone layer | Cluster-validation memo and feature governance notes |

## Sprint 3: Survival Modeling, NLP, and Product Integration

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Build restaurant survival baselines and combine them with healthy-food white-space logic | Build the realistic NLP pipeline and connect the UI to live outputs | Keep the end-to-end pipeline coherent and report-ready |
| Required work | Use official NYC licensing data as the primary restaurant universe; fit Cox PH and Random Survival Forest baselines; combine regime features with competition and rent burden; define the first healthy-food opening score with subtype-level gaps; if time allows, prototype a small CPU-friendly ranking layer; expose `/predict/trajectory` and `/predict/cmf` endpoints | Use a Gemini Flash or Flash-Lite model to generate silver labels for Yelp reviews; retain only high-confidence examples after audit; manually annotate 200 to 300 gold examples for held-out evaluation; aggregate those labels into healthy-demand features instead of training a custom transformer; build white-space and competitor features; connect frontend components to backend responses | Review survival target construction; sign off on the healthy-food opening-score assumptions; run end-to-end QA across ETL, model, API, and Streamlit layers; maintain the bug tracker |
| Deliverables | Survival models, healthy-food score prototype, subtype-aware recommendations, and backend endpoints | NLP models, demand features, and frontend integration | Approved end-to-end prototype and issue log |

## Sprint 4: Backtesting, Robustness, and Final Packaging

| Area | Harsh & Siqi (Backend / ML) | Tony & Amanda (Frontend / Data) | Catherine (Lead / Integration) |
| :---- | :---- | :---- | :---- |
| Main goals | Prove the modeling story with time-aware evaluation | Make the demo stable and interpretable | Finalize the report and presentation package |
| Required work | Run blocked or rolling temporal backtests using the audited cutoff; perform ablations on major feature families; generate figures for survival, clustering, and healthy-food ranking performance; document residual risks; report ranking metrics if the learned ranker is implemented | Fix integration bugs; add a data-freshness note to the UI; make the frontend shortlist-first rather than map-first; build evidence cards with healthy supply-gap summaries, risk flags, and confidence; write the data pipeline and NLP sections of the report | Compile all report sections; write the executive summary and conclusion; ensure citations, figures, and tables are consistent; lead rehearsals and lock the final presentation flow |
| Deliverables | Final evaluation package and model artifacts | Stable demo and polished UI | Final report draft, presentation deck, and submission checklist |

## Non-Negotiable Project Rules

- Do not reintroduce Google Trends.
- Do not commit to Reddit as a core signal until the sparsity audit is complete.
- Do not assume Yelp is the restaurant universe until the NYC coverage audit is done.
- Do not use a random train/test split for the headline result.
- If historical depth is weak for a dataset, downgrade it to a static covariate or replace it with the documented fallback.
- Do not frame the product as a generic restaurant recommender; keep the healthy-food white-space use case explicit.
