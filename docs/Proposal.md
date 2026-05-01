# Written Proposal

## NYC Restaurant Intelligence Platform

Updated: April 30, 2026
Team: Catherine · Harsh · Tony · Siqi · Amanda
Repo: <https://github.com/Amanda-dong/CS473-FML>

## Problem

Independent restaurant owners in NYC make high-cost neighborhood decisions with
far less market intelligence than large chains. Chain operators can buy
foot-traffic studies, demographic analysis, and competitive mapping before
signing a lease; an independent operator often relies on intuition, anecdotal
knowledge, and a quick search across review sites.

This project closes part of that gap by building a restaurant opportunity
platform for NYC with a specific commercial use case. The core product question
is:

> Where in NYC should a merchant open a healthier fast-casual restaurant, and
> which underserved zones currently have the strongest combination of unmet
> healthy-food demand, survivable local economics, and manageable competition?

The project is interesting because restaurant success is a **timing** problem as
much as a location problem. Opening in the right neighborhood after that
neighborhood has already peaked can be as damaging as opening in the wrong
neighborhood altogether. Our contribution is to operationalize that timing
signal with a defensible, time-aware data pipeline rather than a static
neighborhood score, while focusing on a distinctive merchant problem: identifying
healthy-food white space in dense urban micro-markets such as campus-adjacent
lunch corridors.

## Data Strategy (Locked April 1, 2026)

The project is reframed around stable public data and audit-first methodology.

**Core datasets** (all integrated):

- NYC DOB building permits — renovation and development velocity
- NYC DCWP/DCA Legally Operating Businesses — official business-license activity
- NYC DOHMH restaurant inspection results — quality and churn-adjacent signals
- U.S. Census ACS 5-year estimates — demographics and housing context
- NYC PLUTO / MapPLUTO — lot-level land-use and commercial-value proxies
- Inside Airbnb — short-term-rental pressure (subject to historical availability)
- Citi Bike trip and station data — mobility and walkability signals
- Yelp Fusion API + Yelp Open Dataset — only after explicit NYC coverage audit
- NYC 311 complaints — official social signal (Reddit fallback)

Reference geometry (NTA boundaries, Community District boundaries) is treated
as join infrastructure, not as a separately modeled dataset. **Google Trends is
removed from the plan.** Reddit is used only as a coarse-geography mention
signal, with NYC 311 as the documented fallback.

## Methods

The platform uses three coordinated modeling components plus a front-loaded
data audit.

### Feature Families and Schema Governance

The feature schema is treated as a first-class artifact rather than something
improvised late in the pipeline.

Canonical identifiers:

- `nta_id` — source-level neighborhood key
- `zone_id` — final recommendation geography after aggregation to micro-zones
- `restaurant_id` — business-history key
- `review_id` — review-label key
- `time_key` — canonical derived year field for model tables

The processed zone-year matrix
(`data/processed/feature_matrix.parquet`) is the canonical scoring table:
**726 rows × 49 columns** (identifiers plus engineered numeric features).
Authoritative column definitions live in `docs/DataDictionary.md` and the parquet
header on disk; this proposal does not freeze the column list because it
evolves with ETL.

Representative columns include `zone_id`, `time_key`, `license_velocity`,
`net_opens`, `net_closes`, `trip_count`, `station_count`, `healthy_food_share`,
`inspection_grade_avg`, `median_income_static`, `rent_pressure`,
`mean_assessed_value`, `target`, `label_quality`.

Exact model input/output contracts are documented in `docs/ModelInterfaces.md`.

### 1. Neighborhood Phase Discovery

We do not assume NYC neighborhoods come with pre-labeled gentrification phases.
Instead, we construct neighborhood-year feature vectors and use unsupervised
learning to discover phase structure directly from the data.

- Build a neighborhood panel with lagged and normalized features for permits,
  licenses, inspections, demographics, commercial-value proxies, mobility, and
  housing-pressure signals.
- Run k-means and Gaussian Mixture Models as the primary phase-discovery
  methods.
- Choose the final clustering specification using cluster stability,
  silhouette-style separation, and interpretability.
- Label clusters post-hoc based on centroids and temporal trajectories — for
  example, a cluster with rising permit velocity, rising commercial values, and
  business turnover may be interpreted as a gentrifying regime.
- Validate discovered clusters by spot-checking assignments against NYU Furman
  Center "State of NYC Housing and Neighborhoods" reports.

This layer provides macro neighborhood context, not the final recommendation
unit.

### 2. Restaurant Survival Modeling

The survival component remains central, but the data source priority changes.
Official NYC business-license activity is the primary signal for openings,
status changes, and expiration timing. Yelp is treated as secondary enrichment.

- Use official NYC licensing records as the primary restaurant universe.
- Derive opening and closure proxies from status, issuance, and expiration
  behavior where supported by the data.
- Fit Cox Proportional Hazards and Random Survival Forest baselines.
- Feed neighborhood phase features, competition measures, rent proxies, and
  inspection history into the survival model.

This reframing handles right-censored restaurant histories explicitly and uses
official city data rather than incomplete platform coverage. Current
performance: **C-index ≈ 0.80**.

### 3. NLP and Demand Signals

The NLP pipeline avoids unrealistic manual-labeling burden while keeping a
strong ML component.

- Use a Gemini Flash / Flash-Lite model to generate silver sentiment labels
  and short rationales for Yelp review text.
- Keep only high-confidence labels after a small audit pass.
- Manually annotate 200–300 reviews as a held-out gold evaluation set.
- Aggregate the resulting labels directly into healthy-demand and food-mix
  features.
- Keep transformer fine-tuning out of the main plan unless cost pressure
  later forces a lightweight local classifier.

The healthy-food taxonomy (defined up front, see `src/utils/taxonomy.py`):

- healthy fast casual
- salad / bowl concepts
- Mediterranean or grain-bowl concepts
- healthy Indian or South Asian bowl concepts
- vegetarian or vegan grab-and-go
- protein-forward lunch options

This lets the system distinguish between broad healthy-food coverage and
**subtype-level white space**. A district with several popular Mediterranean
bowl chains may still have room for healthy Indian or South Asian fast casual.

Reddit handling is intentionally narrow:

- spaCy NER + a static lookup table of NYC neighborhood names extracts
  location mentions.
- Aggregate Reddit-derived signals at Community District level rather than
  trying to geocode every post to NTA.
- Use a binary "mentioned in the last six months" feature instead of a
  fragile continuous sentiment score.
- If Reddit is too sparse, replace it with NYC 311 complaints under the
  same coarse-grained join strategy.

### 4. Temporal Validation and Backtesting

Temporal alignment is a first-class project risk.

The team ran a one-day audit sprint and recorded per dataset:

- earliest NYC year available
- refresh cadence
- temporal granularity
- spatial granularity
- join key or crosswalk requirement
- fallback if the dataset is incomplete

The full audit lives in `docs/temporal_audit.md`. The current locked window
is **2020–2024** (configurable in `src/config/constants.py`) with rolling
backtests producing the headline numbers in `docs/EvaluationResults.md`. **No
random split is used for the main evaluation.**

### 5. Micro-Zone Recommendation Layer

Neighborhood context alone is too coarse for merchant decisions. A
healthier-food merchant does not choose between borough-sized areas; they
choose between walkable lunch catchments.

The recommendation unit is therefore a **micro-zone**:

- 10-minute walk shed around a campus
- transit-centered lunch corridor
- business-district catchment
- small grid or H3 cell where point coverage allows

The current build ships **137 micro-zones** spanning campus, lunch-corridor,
transit-catchment, and business-district types.

The final score combines two ideas:

- **Healthy-food supply gap** — count of healthy options nearby; ratio of
  healthy options to all quick-service options; subtype saturation; review
  text indicating unmet healthy demand.
- **Merchant viability** — neighborhood regime; restaurant survival risk;
  competition intensity; rent / cost-pressure proxies.

## What Makes The Product Useful

The product is a **shortlist engine**, not a generic exploratory dashboard:

- the user enters a healthy concept subtype and optional constraints
- the system returns the top 5 underserved zones — not every geography at once
- each zone comes with a healthy supply-gap summary, a recommended concept
  subtype, key drivers, risk flags, confidence, and data freshness
- the user can run simple what-if comparisons by changing the concept subtype
  or risk tolerance

This product framing matters because restaurant operators need a
recommendation they can act on, not a screen full of disconnected metrics.

## Why The ML Story Stands Out

The strongest version of this project is not a single model but a layered ML
system:

- unsupervised neighborhood regime discovery
- restaurant survival prediction
- LLM-assisted weak labeling and direct aggregation for healthy-demand signals
- an interpretable ranking layer that combines the above into a healthy-food
  white-space recommendation, with a LambdaMART learning-to-rank head as a
  stretch goal

## Research-Driven Changes Adopted on 2026-04-01 (and shipped)

1. Gentrification labels are discovered with unsupervised clustering and
   validated against external neighborhood references instead of being
   hand-labeled in advance.
2. Reddit is handled with NER and Community District aggregation, with 311
   complaints as the documented fallback.
3. Google Trends is removed from the feature plan.
4. Yelp is audited before being trusted; official NYC licensing data is the
   primary survival backbone.
5. Gemini-generated silver labels are aggregated directly, with no transformer
   fine-tuning in the main plan.
6. A temporal coverage audit determines the final backtesting window.

## Optional Secondary Modeling Path

A hand-built Random Forest baseline remains a useful secondary experiment. It can
be trained on externally sourced neighborhood labels or another downstream
supervised target after the temporal audit stabilizes—without rewriting the core
research story around brittle hand-labeled regimes.

## Status as of April 30, 2026

The implementation is complete across all eight planned stages:

1. Data source audit — 10 ETL modules with real NYC Open Data integrations
2. Canonical neighborhood feature matrix — 726 rows × 49 features
3. Micro-zone layer — 137 zones across four catchment types
4. Phase discovery — k-means + GMM trajectory clusters (k=3 and k=4 evaluated)
5. Survival modeling — Cox PH + RSF; C-index ≈ 0.80
6. NLP labeling and aggregation — Gemini Flash silver labels on the full Yelp corpus
7. Healthy-food white-space ranking — XGBoost scoring + LambdaMART ranker
8. API and Streamlit integration — FastAPI backend, shortlist-first UI

Test suite: **606 passing**.

See `docs/Design.md` for repo structure and division of labor, and
`docs/Sprints.md` for the sprint-by-sprint completion status.
