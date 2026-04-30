# NYC Restaurant Intelligence Platform

## Teammates

- Amanda Dong (`yd2825`) - [GitHub](https://github.com/Amanda-dong)
- Tony Zhao (`sz3822`) - [GitHub](https://github.com/Tonyzsp)
- Harsh Agarwal (`ha2957`) - [GitHub](https://github.com/harshagarwalnyu)
- Siqi Zhu (`sz3950`) - [GitHub](https://github.com/HelenZhutt)
- Catherine Yi (`cgy2014`)- [GitHub](https://github.com/catherinegyi)

Updated: April 26, 2026<br>
Course project · Spring 2026<br>
Team: Catherine, Harsh, Tony, Siqi, Amanda

## What This Project Should Do

This project should help an independent restaurant operator answer one concrete question with as little friction as possible:

Where in NYC should a merchant open a healthier fast-casual restaurant, and which underserved zones should they shortlist first?

The goal is not to build another generic city dashboard. The goal is to produce a recommendation workflow that is actually decision-useful:

- enter a healthy-food concept and a few constraints
- get a ranked shortlist of underserved zones
- see the evidence behind each recommendation
- understand the risk, confidence, and tradeoffs before taking action

## What Makes This Version Better

- Official NYC data is the backbone. Third-party platforms are enrichment layers, not the system of record.
- Neighborhood phases are discovered with unsupervised learning instead of weak hand-labeled classes.
- Restaurant survival is modeled explicitly with time-to-event methods instead of forcing everything into classification.
- Review sentiment is bootstrapped with Gemini-generated labels and aggregated directly, avoiding GPU-heavy local fine-tuning in the main plan.
- The final product is shortlist-first and explanation-first, not map-first.
- The product has a distinctive use case: finding healthy-food white space near campuses, office clusters, and lunch corridors.

## Recommended User Experience

The most useful end-user flow is:

1. User enters a healthy concept subtype such as salad bowls, Mediterranean bowls, healthy Indian or South Asian bowls, protein-heavy lunch, or vegan grab-and-go, plus optional constraints.
2. System returns the top 5 underserved micro-zones, not every geography at once.
3. Each recommendation card shows:
   - overall opportunity rank
   - confidence bucket
   - healthy supply-gap summary
   - recommended concept subtype for that zone
   - key positive drivers
   - key risk flags
   - similar existing restaurants
   - data freshness note
4. The user can click into a scenario view to see what changes if they shift concept subtype, price tier, or risk tolerance.

A map is still useful, but it should support the shortlist instead of being the primary interface. A good motivating example is a campus-adjacent zone like NYU Tandon / MetroTech, where dense lunch demand may coexist with a weak healthy-food mix.

## Core ML Stack

### 1. Neighborhood Phase Discovery

- Build a neighborhood-year panel from permits, licenses, inspections, ACS, PLUTO, mobility, and housing-pressure features.
- Use k-means and Gaussian Mixture Models to discover neighborhood regimes.
- Validate regime assignments against NYU Furman Center neighborhood narratives.

This layer provides macro context, but recommendations should be made at a smaller decision unit than the full neighborhood whenever possible.

### 2. Restaurant Survival Modeling

- Use official NYC business-license activity as the primary restaurant universe.
- Train Cox Proportional Hazards and Random Survival Forest baselines.
- Use neighborhood regime, competition, inspection, and rent-pressure features as covariates.

This layer estimates whether a promising healthy-food gap is also commercially survivable.

### 3. NLP and Demand Signals

- Use a Gemini Flash or Flash-Lite model to generate silver sentiment labels for review text.
- Audit and retain only high-confidence labels.
- Manually label a small gold evaluation set.
- Aggregate the labels directly into healthy-demand features instead of training a custom transformer in the main plan.
- Optional stretch only: distill the labeled data into a lightweight CPU-friendly classifier if the API cost becomes a problem.
- Use Reddit only as a coarse geography mention signal, with NYC 311 complaints as the fallback.

Useful derived features include:

- review share mentioning healthy, fresh, salad, protein, vegan, vegetarian, light, or quick lunch
- complaint or mention density for lack of healthy options
- local food-mix indicators showing dominance of burger, pizza, fried, or dessert-heavy quick-service options
- subtype-specific competition such as Mediterranean-bowl saturation versus healthy Indian or South Asian whitespace

### 4. Final Ranking Layer

- MVP: interpretable healthy-food white-space score from the three components above.
- Stretch goal: learning-to-rank model that orders candidate zones for a healthy concept query.

The score should be subtype-aware, not just category-aware. A zone can have several popular healthy bowl concepts and still be underserved for a different healthy subtype, such as healthy Indian fast casual.

The practical recommendation unit should be a micro-zone such as:

- 10-minute walk sheds around campuses
- transit-centered lunch corridors
- business-district catchments
- small grid or H3 cells if the data quality supports them

## Data Strategy

Primary sources:

- NYC DOB permits
- NYC DCWP/DCA Legally Operating Businesses
- NYC DOHMH restaurant inspection results
- U.S. Census ACS 5-year estimates
- NYC PLUTO / MapPLUTO
- Inside Airbnb, subject to historical coverage limits
- Citi Bike trip or station data

Conditional enrichment sources:

- Yelp Fusion API and Yelp Open Dataset, only after NYC coverage audit
- Reddit neighborhood mentions at Community District level
- NYC 311 complaints as the documented fallback if Reddit is sparse

## Repository Status

The implementation is complete. All eight planned stages have been delivered:

1. **Data source audit** — 10 ETL modules with real NYC Open Data integrations (permits, licenses, inspections, ACS, PLUTO, Citi Bike, Airbnb, Yelp, 311, boundaries)
2. **Canonical neighborhood feature matrix** — 773 zone-year rows, 33 features across all data sources
3. **Micro-zone layer** — 8 campus, lunch-corridor, transit-catchment, and business-district zones across all 5 boroughs
4. **Phase discovery** — k-means and GMM trajectory clustering (k=3 and k=4 evaluated); NTA healthy food-story cluster assignments
5. **Survival modeling** — Cox PH + Random Survival Forest; `survival_model.joblib` trained and evaluated
6. **NLP labeling and aggregation** — Gemini Flash silver labels on full Yelp corpus; zone-level healthy-demand features in `gemini_full_zone_features.csv`
7. **Healthy-food white-space ranking** — XGBoost scoring model + LambdaMART ranker; interpretable CMF score with subtype-gap, survival-risk, and positive-driver explanations
8. **API and Streamlit integration** — FastAPI backend (`/predict/cmf`, `/predict/trajectory`), Streamlit frontend with recommendation cards, map view, and scenario panel

Model artifacts: `data/models/scoring_model.joblib`, `survival_model.joblib`, `ranking_model.joblib`

Evaluation artifacts: `data/processed/backtest_results.parquet`, `ablation_results.parquet`, `docs/EvaluationResults.md`, `docs/CausalMLEvaluationReport.md`

Test suite: 520 tests, all passing (`uv run pytest`)

## Repository Structure

```text
CS473-FML/
├── README.md
├── .env.example
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── geojson/
│   └── models/
├── docs/
├── frontend/
│   ├── app.py
│   ├── components/
│   └── pages/
├── notebooks/
├── scripts/
├── src/
│   ├── api/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── nlp/
│   ├── pipeline/
│   ├── schemas/
│   ├── utils/
│   └── validation/
└── tests/
```

## Setup

Recommended local workflow uses `uv`.

1. Install `uv`.
2. Create a virtual environment:
   - `uv venv`
3. Activate it:
   - `source .venv/bin/activate` on macOS/Linux
   - `.venv\\Scripts\\activate` on Windows
4. Install dependencies:
   - `uv pip install -r requirements.txt`
5. Run the test suite:
   - `uv run pytest`

Note: A `GEMINI_API_KEY` is required for live NLP labeling (`src/nlp/gemini_labels.py`). All other features run without API keys.

## Backend Quick Start

1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Start API server:
   - `bash scripts/run_api.sh`
3. In a separate terminal, run smoke checks:
   - `python scripts/smoke_api.py`

Backend API contract is documented in:

- `docs/api_contract.md`

## Documentation Index

- `docs/Proposal.md`
- `docs/Design.md`
- `docs/Sprints.md`
- `docs/Research.md`
- `docs/api_contract.md`
