# Written Proposal

## NYC Restaurant Intelligence Platform

Updated: April 1, 2026
Team: Catherine · Harsh · Tony · Siqi · Amanda

## Problem

Independent restaurant owners in NYC make high-cost neighborhood decisions with far less market intelligence than large chains. Chain operators can buy foot-traffic studies, demographic analysis, and competitive mapping before signing a lease; an independent operator often relies on intuition, anecdotal knowledge, and a quick search across review sites.

This project aims to close part of that gap by building a restaurant opportunity platform for NYC with a more specific commercial use case. The core product question is:

Where in NYC should a merchant open a healthier fast-casual restaurant, and which underserved zones currently have the strongest combination of unmet healthy-food demand, survivable local economics, and manageable competition?

The project is interesting because restaurant success is a timing problem as much as a location problem. Opening in the right neighborhood after the neighborhood has already peaked can be as damaging as opening in the wrong neighborhood altogether. Our contribution is to operationalize that timing signal with a defensible, time-aware data pipeline rather than a static neighborhood score, while also focusing on a distinctive and useful merchant problem: identifying healthy-food white space in dense urban micro-markets such as campus-adjacent lunch corridors.

## Revised Data Strategy

As of April 1, 2026, the project is being reframed around stable public data and audit-first methodology.

Core datasets:

- NYC DOB building permits for renovation and development velocity
- NYC DCWP/DCA Legally Operating Businesses for official business-license activity
- NYC DOHMH restaurant inspection results for restaurant quality and churn-adjacent signals
- U.S. Census ACS 5-year estimates for demographics and housing context
- NYC PLUTO / MapPLUTO for lot-level land-use and commercial value proxies
- Inside Airbnb for short-term-rental pressure, subject to historical availability
- Citi Bike trip or station data for mobility and walkability signals
- Yelp Fusion API and Yelp Open Dataset, but only after an explicit NYC coverage audit
- Reddit neighborhood mentions at coarse geography, with NYC 311 complaints as the fallback source if Reddit is sparse

Reference geometry such as NTA boundaries and Community District boundaries is treated as join infrastructure, not as a separate modeled dataset. Google Trends is removed from the plan.

## Methods

The platform now uses three coordinated modeling components plus a front-loaded data audit.

### Feature Families and Schema Governance

The project now treats the feature schema as a first-class artifact instead of
something that gets improvised late in the pipeline.

Canonical identifiers:

- `nta_id`: source-level neighborhood key
- `zone_id`: final recommendation geography after aggregation to micro-zones
- `restaurant_id`: business-history key
- `review_id`: review-label key
- `time_key`: canonical derived year field for model tables

Current implemented zone-year matrix columns are documented in `docs/DataDictionary.md`
and currently include:

- `zone_id`
- `time_key`
- `license_velocity`
- `net_opens`
- `net_closes`
- `healthy_review_share`
- `social_buzz`
- `population`
- `median_income`
- `rent_burden`
- `inspection_grade_avg`
- `restaurant_count`
- `rent_pressure`
- `mean_assessed_value`

The exact model input and output contracts are documented in `docs/ModelInterfaces.md`.
Those two files should be treated as the detailed reference when the proposal text
is too high-level for implementation work.

### 1. Neighborhood Phase Discovery

We will no longer assume that NYC neighborhoods come with clean pre-labeled gentrification phases. Instead, we will construct neighborhood-year feature vectors and use unsupervised learning to discover phase structure directly from the data.

Planned approach:

- Build a neighborhood panel with lagged and normalized features for permits, licenses, inspections, demographics, commercial-value proxies, mobility, and housing-pressure signals
- Run k-means and Gaussian Mixture Models as the primary phase-discovery methods
- Choose the final clustering specification using cluster stability, silhouette-style separation metrics, and interpretability
- Label clusters post-hoc based on their centroids and temporal trajectories, for example a cluster with rising permit velocity, rising commercial values, and business turnover may be interpreted as a gentrifying regime
- Validate discovered clusters by spot-checking assignments against NYU Furman Center "State of NYC Housing and Neighborhoods" reports, which provide narrative evidence of neighborhood change

This change removes the need to hand-label all neighborhoods up front. If the course ultimately requires a supervised classifier, we will treat Furman Center or Urban Displacement Project references as external labels and train a downstream classifier from those labels rather than inventing labels ourselves. This layer is meant to provide macro neighborhood context, not the final recommendation unit by itself.

### 2. Restaurant Survival Modeling

The survival component remains central, but the data source priority changes. Official NYC business-license activity becomes the primary signal for restaurant openings, status changes, and expiration timing. Yelp is treated as a secondary enrichment source, not the main survival dataset.

Planned approach:

- Use official NYC licensing records as the primary restaurant universe
- Derive opening and closure proxies from status, issuance, and expiration behavior where supported by the data
- Fit Cox Proportional Hazards and Random Survival Forest baselines
- Feed neighborhood phase features, competition measures, rent proxies, and inspection history into the survival model

This reframing is stronger methodologically because right-censored restaurant histories are handled explicitly and the base records come from official city data rather than incomplete platform coverage. It also prevents the system from recommending an obviously underserved zone that is still structurally hard for merchants to survive in.

### 3. NLP and Demand Signals

The NLP pipeline is revised to avoid an unrealistic manual-labeling burden while still keeping a strong ML component.

Planned approach:

- Use a Gemini Flash or Flash-Lite model to generate silver sentiment labels and short rationales for Yelp review text
- Keep only high-confidence labels after a small audit pass
- Manually annotate 200 to 300 reviews as a held-out gold evaluation set
- Aggregate the resulting labels directly into healthy-demand and food-mix features
- Keep transformer fine-tuning out of the main plan unless later scaling or cost pressure makes a lightweight local classifier necessary

This is still a legitimate weak-supervision setup, but it avoids spending project time on GPU-heavy fine-tuning. The LLM is used for offline labeling, while the core ML contribution stays in clustering, survival analysis, and temporal evaluation.

The project should define a simple healthy-food taxonomy up front. For example:

- healthy fast casual
- salad / bowl concepts
- Mediterranean or grain-bowl concepts
- healthy Indian or South Asian bowl concepts
- vegetarian or vegan grab-and-go
- protein-forward lunch options

This lets the system estimate both healthy-food demand and healthy-food supply gaps rather than treating all restaurants as interchangeable. It also lets the system distinguish between broad healthy-food coverage and subtype-level white space. For example, a district with several popular Mediterranean bowl chains may still have room for healthy Indian or South Asian fast casual.

Reddit is also narrowed:

- Use spaCy NER plus a static lookup table of NYC neighborhood names to extract location mentions
- Aggregate Reddit-derived signals at Community District level instead of trying to geocode every post to NTA
- Use a binary presence feature such as "mentioned in the last six months" rather than a fragile continuous sentiment score
- If Reddit is too sparse, replace it with NYC 311 complaints and keep the same coarse-grained join strategy

### 4. Temporal Validation and Backtesting

Temporal alignment is now treated as a first-class project risk.

Before finalizing the training window, the team will run a one-day audit sprint and record for each dataset:

- earliest NYC year available
- refresh cadence
- temporal granularity
- spatial granularity
- join key or crosswalk requirement
- fallback if the dataset is incomplete

The final train and test cutoff will be determined by the most constraining high-value dataset, not by an idealized date range. If one source only supports a shorter history, it becomes either the limiting window or a static covariate. A realistic fallback is a 2016 to 2020 training window with 2021 to 2022 held out for temporal backtesting.

No random split will be used for the main evaluation. The primary validation story is blocked or rolling temporal backtesting.

### 5. Micro-Zone Recommendation Layer

Neighborhood context alone is too coarse for merchant decisions. A healthier-food merchant does not choose between entire borough-sized areas; they choose between walkable lunch catchments.

The recommendation unit should therefore be a micro-zone such as:

- a 10-minute walk shed around a campus
- a transit-centered lunch corridor
- a business-district catchment
- a small grid or H3 cell if point coverage is good enough

This allows the system to capture cases like a university-adjacent district that has strong lunch demand but an unhealthy local option mix dominated by burgers, pizza, and chains.

The final score should combine two ideas:

- healthy-food supply gap
- merchant viability

Healthy-food supply gap should reflect:

- how many healthy options already exist nearby
- the ratio of healthy options to all quick-service options
- whether one healthy subtype is saturated while another is missing
- whether local review text or social signals indicate a need for fresher or healthier lunch choices

Merchant viability should reflect:

- neighborhood regime
- restaurant survival risk
- competition intensity
- rent or cost pressure proxies

## What Makes The Product Useful

The project should not end as a generic exploratory dashboard. The most useful end-user experience is a shortlist engine:

- the user enters a healthy concept subtype and optional constraints
- the system returns the top 5 underserved zones rather than overwhelming the user with every geography at once
- each zone comes with a healthy supply-gap summary, a recommended concept subtype, key drivers, risk flags, confidence, and data freshness
- the user can run simple what-if comparisons by changing the concept subtype or risk tolerance

This product framing matters because restaurant operators need a recommendation they can act on, not a screen full of disconnected metrics.

## Why The ML Story Stands Out

The strongest version of this project is not a single model but a layered ML system:

- unsupervised neighborhood regime discovery
- restaurant survival prediction
- LLM-assisted weak labeling and direct aggregation for healthy-demand signals
- an interpretable ranking layer that combines the above into a healthy-food white-space recommendation

If time allows, the final CMF layer can move beyond a hand-weighted score into a small CPU-friendly ranking experiment. The interpretable weighted score remains the main plan.

## Research-Driven Changes Adopted on 2026-04-01

The proposal now explicitly adopts the following choices:

1. Gentrification labels are discovered with unsupervised clustering and validated against external neighborhood references instead of being fully hand-labeled in advance.
2. Reddit is handled with NER and Community District aggregation, with 311 complaints as the documented fallback.
3. Google Trends is removed from the feature plan because it is noisy, unofficial, and poorly aligned to neighborhood geography.
4. Yelp is audited before being trusted; official NYC licensing data becomes the primary survival backbone.
5. Gemini-generated silver labels are aggregated directly, with no transformer fine-tuning in the main plan.
6. A temporal coverage audit determines the final backtesting window before the proposal is locked.

## Course-Project Framing

If a from-scratch model implementation is still required for the course, the hand-built Random Forest remains a valid secondary workstream. It can be trained on externally sourced neighborhood labels or on another downstream supervised target after the audit is complete. That preserves the course requirement without forcing the main research question into a weak supervised-label setup.
