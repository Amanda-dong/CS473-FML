# NYC Restaurant Intelligence Platform - Design Document

Updated: April 1, 2026  
Team: Catherine · Harsh · Tony · Siqi · Amanda  
Repo: github.com/Amanda-dong/CS473-FML

## 1. Repository Structure

Current scaffold:

```text
CS473-FML/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── geojson/
├── docs/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── nlp/
│   ├── api/
│   └── validation/
├── frontend/
│   ├── app.py
│   └── components/
├── notebooks/
└── tests/
```

Design intent by area:

- `data/raw/`: date-stamped source extracts and audit snapshots; never treated as hand-edited working data
- `data/processed/`: canonical cleaned tables that feed feature generation
- `data/geojson/`: boundary files and crosswalk assets for NTA and Community District joins
- `src/data/`: ETL modules for each approved data source
- `src/features/`: feature assembly, competition scores, rent proxies, and neighborhood panel construction
- `src/models/`: phase discovery, survival modeling, and concept-market-fit scoring
- `src/nlp/`: review pseudo-labeling, sentiment modeling, and neighborhood mention extraction
- `src/validation/`: temporal backtesting, ablations, and robustness checks
- `frontend/`: Streamlit prototype for interactive neighborhood exploration

The product target is no longer a generic restaurant recommender. It is an underserved healthy-food locator for merchants, with special emphasis on campus-adjacent zones, office lunch corridors, and similar walkable micro-markets.

## 2. Research-Driven Architecture Decisions

| Area | Updated decision | Why it is better |
| :---- | :---- | :---- |
| Neighborhood phases | Replace hand-labeled supervised phase classification with unsupervised phase discovery using k-means and GMM | Removes the weakest assumption in the original proposal and lets the data define regimes before interpretation |
| Cluster validation | Validate discovered clusters against NYU Furman Center neighborhood reports; use Urban Displacement Project or Furman references only if a supervised target is required | Keeps external evidence in the loop without forcing noisy labels up front |
| Reddit geography | Extract neighborhood mentions with spaCy NER and a static NYC lookup table; aggregate at Community District level | More tractable than trying to geocode sparse social text to 195 NTAs |
| Reddit feature shape | Use a binary recent-mention signal, not a continuous sentiment score | Lower variance and easier to justify with sparse data |
| Social-data fallback | If Reddit is too sparse, replace it with NYC 311 complaint data | 311 is official, geocoded, and easier to aggregate consistently |
| Google Trends | Remove from the project plan entirely | `pytrends` is unofficial and neighborhood-level signal quality is weak |
| Yelp usage | Audit NYC coverage before relying on Yelp; treat Yelp as enrichment, not the system of record | Prevents the survival model from being built on incomplete platform coverage |
| Survival backbone | Use official NYC business-license activity as the primary restaurant universe | Stronger coverage and better temporal completeness than review-platform data alone |
| Sentiment labels | Generate silver labels with Gemini, audit a small gold set, and aggregate labels directly without transformer fine-tuning in the core plan | Preserves the NLP signal while keeping compute costs low |
| Temporal validation | Run a data audit first and set the train/test cutoff based on the constraining dataset | Avoids promising a backtest window the data cannot actually support |

## 3. Data Model and Join Strategy

Primary analytical units:

- neighborhood-year panel keyed by NTA and year for the macro context features
- micro-zone keyed by walk-shed, corridor, or small spatial cell for final recommendations

Secondary units:

- Community District month or quarter for Reddit or 311 social-signal aggregation
- restaurant-level histories for survival modeling
- review-level records for NLP training

Join strategy:

- Maintain a clean distinction between point data, polygon data, and text-derived signals
- Join geocoded operational data to NTA boundaries for the main neighborhood panel
- Build a second layer of micro-zones around campuses, transit stops, or lunch corridors for the final recommendation surface
- Aggregate Reddit or 311 features first at Community District level, then map to NTAs through a documented crosswalk or area-based aggregation
- Treat any source without trustworthy historical depth as a static covariate instead of pretending it is a full time series

### 3.1 Exact Schema References

The project now has two implementation-level companion docs:

- `docs/DataDictionary.md`: exact raw and derived dataset schemas, phase-by-phase column usage, and current implementation status
- `docs/ModelInterfaces.md`: exact model inputs, outputs, diagnostics, runtime behavior, and algorithm rationale

Those two files should be treated as the schema and model source of truth when the
team needs exact column names instead of high-level design prose.

## 4. Core Modeling Plan

### 4.1 Phase Discovery

- Build lagged neighborhood features from permits, licenses, inspections, ACS, PLUTO, mobility, and housing-pressure data
- Run k-means and Gaussian Mixture Models
- Compare candidate cluster counts with separation and stability diagnostics
- Interpret cluster centroids post-hoc and validate against known neighborhood narratives

### 4.2 Survival Modeling

- Construct restaurant-level histories from official NYC licensing data
- Add neighborhood context from the phase-discovery module
- Train Cox Proportional Hazards and Random Survival Forest baselines
- Use Yelp only when it contributes real incremental coverage or text features

This layer should be concept-aware where possible. A healthy fast-casual concept does not compete against all restaurants equally; it competes most directly within lunch-oriented and adjacent healthier categories.

### 4.3 NLP and Demand Signals

- Use a Gemini Flash or Flash-Lite model to generate silver labels for Yelp review text
- Audit and retain only high-confidence labels
- Manually label a small gold test set for honest evaluation
- Aggregate those labels into healthy-demand and food-mix features
- Keep local transformer fine-tuning as optional only, not part of the baseline implementation
- Use spaCy NER for neighborhood mention extraction from Reddit
- Fall back to NYC 311 complaint counts if Reddit coverage is inadequate

Recommended text labels:

- healthy / fresh / light
- salad / bowl / protein-forward
- healthy Indian / South Asian
- Mediterranean / grain-bowl
- vegan / vegetarian
- quick lunch / grab-and-go
- unhealthy-dominant local mix such as burger / pizza / fried-heavy

### 4.4 Final Score

The final score should combine:

- healthy-food supply gap
- concept-subtype gap
- merchant viability
- neighborhood phase or trajectory regime
- competition saturation penalty

Recommended implementation:

- baseline: interpretable weighted score for transparency
- stretch goal: small CPU-friendly ranking model that orders neighborhoods within a concept query

Healthy-food supply gap should be computed at the micro-zone level and should reflect:

- count of healthy options nearby
- ratio of healthy options to all quick-service options
- local review evidence of unmet healthy demand
- food-mix imbalance toward burgers, pizza, fried, or dessert-heavy chains
- subtype-level white space inside the healthy category, for example Mediterranean saturation but healthy Indian under-supply

Merchant viability should reflect:

- local survival odds
- rent pressure
- neighborhood regime
- congestion or competitive intensity

The score should be framed as a ranking aid, not as a guaranteed forecast of business success.

## 5. Product Experience Requirements

The product should optimize for decision speed, not dashboard complexity.

Required UX principles:

- shortlist-first output: show the best 5 neighborhoods before showing the full map
- shortlist-first output: show the best 5 underserved healthy-food zones before showing the full map
- explanation-first output: every recommendation must expose key positive drivers and key risks
- confidence-first output: display a confidence bucket or uncertainty band alongside the score
- scenario testing: allow users to change healthy concept subtype, price tier, or risk tolerance and immediately compare results
- freshness visibility: show when each source was last refreshed

Minimum recommendation card fields:

- zone name or walk-shed label
- rank or score
- confidence bucket
- healthy supply-gap summary
- recommended concept subtype
- top 3 positive factors
- top 3 risks
- comparable restaurant context
- data freshness note

## 6. Evaluation Metrics That Matter

The project should evaluate each modeling layer with the right metric instead of forcing a single accuracy number.

- clustering: stability and interpretability, not only silhouette
- survival: concordance index and calibration-style checks where feasible
- NLP: agreement between Gemini labels and the manually labeled gold set, plus stability of the derived neighborhood aggregates
- final ranking: top-k usefulness metrics such as NDCG@k or recall@k on held-out periods if the ranking layer is implemented
- product relevance: manual case-study checks on obvious candidate zones such as campus or office lunch districts

## 7. Engineering and Reproducibility Standards

- Use `uv` for Python environment management and command execution instead of `pip`-first setup
- Keep raw source pulls immutable and date-stamped
- Maintain a dataset audit sheet with source URL, refresh cadence, earliest year, spatial granularity, and fallback
- Avoid random train/test splits for the main modeling story; use blocked or rolling temporal validation
- Keep third-party API sources optional and clearly separated from official public datasets
- Refresh dependencies to remove `pytrends` and add planned NLP utilities such as spaCy plus the Gemini labeling client as implementation begins
- Keep the baseline implementation CPU-friendly; avoid GPU-heavy custom model training unless it becomes a clearly justified stretch goal

## 8. Division of Labor

| Member | Role | Updated responsibilities |
| :---- | :---- | :---- |
| **Harsh & Siqi** | Backend / ML Lead | phase discovery experiments, survival models, CMF scoring, API contracts, temporal evaluation logic |
| **Tony & Amanda** | Frontend / Data | source audits, ETL pipelines, Yelp coverage audit, Reddit NER or 311 fallback pipeline, Streamlit integration |
| **Catherine** | Project Lead | data dictionary, crosswalk governance, Furman-based validation rubric, temporal split approval, report integration, presentation |

## 9. Stub Code Status

The repository is no longer just a bare scaffold. The current state is:

- ETL schemas are defined for every planned source
- real fetch/transform paths exist for `permits`, `licenses`, `inspections`, `pluto`, `reddit`, and `complaints_311`
- local-file loaders exist for `acs` and `yelp`
- `airbnb`, `citibike`, and `boundaries` still need real loaders
- the zone-year feature builder is implemented and currently wires in `licenses`, `pluto`, `yelp`, `reddit`, `acs`, and `inspections`
- survival, clustering, NLP aggregation, explainability, scoring, ranking, API, and Streamlit layers all have working code paths
- some documented sources are still only partially connected to downstream features, especially `permits`, `citibike`, `airbnb`, `311`, and the boundary geometry layer

The implementation order should still follow the revised design above:

1. data audit and source viability
2. canonical neighborhood feature matrix
3. micro-zone layer for campuses and lunch corridors
4. phase discovery
5. survival modeling
6. NLP enrichment
7. UI and API integration
