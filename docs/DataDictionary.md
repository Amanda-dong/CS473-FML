# Data Dictionary and Exact Dataset Schemas

Updated: April 3, 2026

This document is the schema-level source of truth for the project.
It covers:

- raw source datasets
- canonical transformed tables
- derived feature tables
- labels and targets
- which phase uses each dataset

The goal is to remove ambiguity. When a model says it uses "licenses" or
"review features," this document spells out the exact columns that means.

## 1. Shared Naming Rules

These identifiers appear throughout the pipeline:

- `nta_id`: NYC Neighborhood Tabulation Area code such as `BK09`, `MN22`, `QN70`
- `zone_id`: the recommendation unit after NTA-to-microzone aggregation
- `community_district`: coarse social-signal geography used for Reddit and 311
- `restaurant_id`: business or restaurant key used across operational tables
- `review_id`: review-level key used in labeling and NLP
- `time_key`: canonical derived time field, usually a year

Important distinction:

- `nta_id` is a source- or feature-engineering geography
- `zone_id` is the recommendation geography

## 2. Phase Map

| Phase | Main table(s) | Exact columns / contract | Status in repo |
| :---- | :---- | :---- | :---- |
| Source audit | `DatasetAuditRow` | `name`, `owner`, `spatial_unit`, `time_grain`, `earliest_year`, `status`, `notes` | implemented |
| Raw ETL | source-specific tables below | fixed canonical schemas per source | mixed: some real ETL, some loaders, some stubs |
| Zone-year feature engineering | zone-year matrix | `zone_id`, `time_key`, joined feature columns listed below | implemented |
| Neighborhood phase discovery | numeric slice of zone-year matrix | all numeric columns from phase matrix | implemented |
| Survival modeling | restaurant history table | exact columns listed below | implemented |
| NLP weak labeling | review-label cache | exact label columns listed below | implemented |
| Review aggregation | zone-time review features | exact aggregate columns listed below | implemented |
| Scoring / ranking | `feature_matrix.parquet` + `target` | all feature columns except dropped identifiers | implemented |

## 3. Raw Source Datasets

This section lists the canonical target schema for each source module.
If a loader currently reads a local file without coercing columns, the table
below still defines the schema that downstream code is supposed to standardize to.

### 3.1 `permits`

Purpose:

- proxy for renovation, construction, and physical change velocity
- intended for neighborhood phase discovery

Current code path:

- `src/data/etl_permits.py`

Canonical columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `permit_date` | currently stored as stringified year after aggregation | retained as the temporal source field |
| `nta_id` | spatial key; currently derived from `communityboard` field name, though this is semantically imperfect | join into neighborhood-level panel |
| `permit_type` | permit subtype from DOB | separate major permit activity families |
| `job_count` | aggregated count of jobs in that `(nta_id, year, permit_type)` cell | velocity feature source |

Phase usage:

- planned for neighborhood phase discovery
- ETL implemented, but not yet wired into `build_zone_year_matrix()`

### 3.2 `licenses`

Purpose:

- official operating-business activity
- used for macro vitality signals and restaurant histories

Current code path:

- `src/data/etl_licenses.py`

Exact transformed columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `event_date` | parsed datetime from `license_creation_date` | event ordering |
| `restaurant_id` | `business_unique_id` | business history key |
| `license_status` | status string such as active, issued, inactive, revoked, expired | open / close proxy |
| `nta_id` | NTA code | geography join key |
| `category` | business category | filtering and later concept logic |

Phase usage:

- phase discovery
- survival history construction
- composite ground truth

Status:

- real ETL implemented and used downstream

### 3.3 `inspections`

Purpose:

- restaurant quality and failure-risk signal
- restaurant-level health signal

Current code path:

- `src/data/etl_inspections.py`

Exact transformed columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `inspection_date` | parsed datetime | event timing |
| `restaurant_id` | `camis` | restaurant join key |
| `grade` | inspection grade, filled to `N` when missing | quality proxy |
| `critical_flag` | violation criticality flag | severity context |
| `nta_id` | mapped from zipcode, with borough fallback | geography join key |
| `cuisine_type` | cuisine description | restaurant subtype context |
| `zipcode` | raw zipcode string | traceability and fallback mapping |

Phase usage:

- phase discovery
- survival modeling
- composite ground truth

Status:

- real ETL implemented and used downstream

### 3.4 `acs`

Purpose:

- stable demographic and housing context

Current code path:

- `src/data/etl_acs.py`

Canonical columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `year` | analysis year | temporal join |
| `nta_id` | NTA code | geography join |
| `median_income` | area median income | price-tier fit and macro context |
| `population` | area population | demand denominator / scale |
| `rent_burden` | housing-rent burden | pressure / affordability context |

Loader note:

- current `run_etl()` simply reads the local file pointed to by `ACS_DATA_PATH`
- the local file therefore must already be standardized to these columns

Phase usage:

- zone-year matrix
- phase discovery
- final scoring features

### 3.5 `pluto`

Purpose:

- built-environment and commercial-cost proxy

Current code path:

- `src/data/etl_pluto.py`

Exact transformed columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `year` | currently mapped from `yearbuilt`; this is a property attribute, not a panel year | retained for traceability, but treated carefully |
| `nta_id` | mapped from zipcode or borough fallback | geography join |
| `commercial_sqft` | commercial area from `comarea` | commercial-intensity proxy |
| `mixed_use_ratio` | `commercial_sqft / bldgarea` when available | mixed-use intensity |
| `assessed_value` | total assessed value from `assesstot` | rent-pressure proxy |

Phase usage:

- zone-year matrix via rent features
- survival zone covariates

Status:

- real ETL implemented and used downstream

### 3.6 `citibike`

Purpose:

- mobility, walkability, and lunch-traffic proxy

Current code path:

- `src/data/etl_citibike.py`

Canonical columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `year` | analysis year | temporal join |
| `nta_id` | NTA code | geography join |
| `trip_count` | annual trips | foot-traffic proxy |
| `station_count` | station count | accessibility proxy |

Phase usage:

- planned for phase discovery and final scoring

Status:

- schema defined, real loader not yet implemented

### 3.7 `airbnb`

Purpose:

- housing-pressure enrichment

Current code path:

- `src/data/etl_airbnb.py`

Canonical columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `year` | analysis year | temporal join |
| `nta_id` | NTA code | geography join |
| `listing_count` | short-term rental inventory | pressure signal |
| `entire_home_ratio` | share of entire-home listings | housing displacement proxy |

Phase usage:

- planned for phase discovery

Status:

- schema defined, real loader not yet implemented

### 3.8 `yelp`

Purpose:

- review text and rating enrichment
- review-level NLP input

Current code path:

- `src/data/etl_yelp.py`

Canonical columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `review_date` | review timestamp | year-level aggregation |
| `business_id` | Yelp business key | source traceability |
| `restaurant_id` | local join key when aligned to internal business IDs | downstream joins |
| `rating` | review star rating | quality / review outcome signal |
| `review_text` | raw review text | NLP weak labeling input |

Loader note:

- current `run_etl()` reads the local file pointed to by `YELP_DATA_PATH`
- the file therefore must be normalized to the canonical columns above before
  it can be treated as the project's standard Yelp table

Phase usage:

- review NLP
- demand features
- composite ground truth

### 3.9 `reddit`

Purpose:

- coarse social mention signal

Current code path:

- `src/data/etl_reddit.py`

Exact transformed columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `month` | `%Y-%m` string | coarse time key |
| `community_district` | matched neighborhood fragment or `Unknown` | coarse geography |
| `mention_text` | truncated post title / text | inspection and QA |
| `subreddit` | source subreddit | provenance |

Phase usage:

- planned social-demand enrichment

Status:

- real fetch path exists, but downstream conversion to `social_buzz` in the
  zone-year matrix is still a placeholder

### 3.10 `complaints_311`

Purpose:

- official fallback when Reddit is sparse

Current code path:

- `src/data/etl_311.py`

Exact transformed columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `month` | `%Y-%m` string derived from complaint date | coarse time key |
| `community_district` | complaint geography | coarse spatial key |
| `complaint_type` | complaint category | complaint-family filtering |
| `count` | monthly complaint count | aggregate signal |

Phase usage:

- planned social-demand fallback

Status:

- real ETL implemented, not yet wired into demand feature builder

### 3.11 `boundaries`

Purpose:

- geometry layer for NTAs, community districts, and microzones

Current code path:

- `src/data/etl_boundaries.py`

Canonical columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `zone_id` | geometry identifier | join key |
| `zone_type` | geometry family | UI and zone filtering |
| `geometry_wkt` | Well-Known Text geometry | mapping / joins |

Phase usage:

- micro-zone geometry and frontend mapping

Status:

- schema defined, real loader not yet implemented

### 3.12 `google_trends`

Purpose:

- historical placeholder only

Current code path:

- `src/data/etl_google_trends.py`

Columns:

| Column | Type / meaning | Why it exists |
| :---- | :---- | :---- |
| `week` | weekly bucket | historical placeholder |
| `term` | query term | historical placeholder |
| `interest` | trend value | historical placeholder |

Status:

- deprecated
- explicitly excluded from active feature engineering

## 4. Audit Inventory Schema

The source audit endpoint uses `DatasetAuditRow` from `src/schemas/datasets.py`.

Exact columns:

| Column | Meaning |
| :---- | :---- |
| `name` | dataset name |
| `owner` | team owner |
| `spatial_unit` | main geography |
| `time_grain` | main temporal grain |
| `earliest_year` | earliest validated year when known |
| `status` | planned / active / deprecated |
| `notes` | free-text caveats |

Why it matters:

- forces explicit provenance and fallback planning before modeling

## 5. Derived Feature Tables

### 5.1 License velocity features

Produced by:

- `build_license_velocity_features()`

Exact columns:

- `zone_id`
- `time_key`
- `license_velocity`
- `net_opens`
- `net_closes`

Source columns used:

- `event_date`
- `license_status`
- `nta_id`

### 5.2 Rent trajectory features

Produced by:

- `build_rent_trajectory_features()`

Exact columns:

- `zone_id`
- `rent_pressure`
- `mean_assessed_value`

Source columns used:

- `nta_id`
- `assessed_value`
- `commercial_sqft`

### 5.3 Demand features

Produced by:

- `build_demand_features()`

Exact columns:

- `zone_id`
- `time_key`
- `healthy_review_share`
- `social_buzz`

Current note:

- `healthy_review_share` is implemented
- `social_buzz` exists in the schema, but the current Reddit-to-zone conversion
  path in `_prepare_social_signals()` still returns an empty placeholder frame

### 5.4 Zone-year feature matrix

Produced by:

- `build_zone_year_matrix()`

Current implemented joined columns:

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

Important note:

- this is the current implemented column set from the builder
- additional planned source families such as permits, Citi Bike, Airbnb, and 311
  are not yet joined into this matrix

### 5.5 Restaurant history for survival

Produced by:

- `build_real_restaurant_history()`

Exact columns:

- `restaurant_id`
- `zone_id`
- `cuisine_type`
- `duration_days`
- `event_observed`
- `inspection_grade_numeric`
- `rent_pressure`
- `competition_score`
- `transit_access`

Why the table exists:

- converts event logs into a right-censored survival dataset

### 5.6 Gemini label cache

Stored at:

- `data/processed/gemini_labels.parquet`

Exact columns:

- `review_id`
- `sentiment`
- `concept_subtype`
- `confidence`
- `rationale`

### 5.7 Aggregated review labels

Produced by:

- `aggregate_review_labels()`

Core output columns:

- `zone_id`
- `time_key`
- `healthy_review_share`
- `subtype_gap`
- `dominant_subtype`

Optional output columns:

- `frac_positive`
- `frac_neutral`
- `frac_negative`
- `topic_0_share`, `topic_1_share`, ..., `topic_{k-1}_share`

### 5.8 Topic-distribution table

Produced by:

- `topic_distribution_per_zone()`

Exact output pattern:

- `zone_id`
- `topic_0_share`
- `topic_1_share`
- ...
- `topic_{k-1}_share`

### 5.9 Zone embedding-feature table

Produced by:

- `compute_zone_embedding_features()`

Exact output pattern:

- `zone_id`
- `embedding_diversity`
- `topic_0_share`, `topic_1_share`, ..., `topic_{k-1}_share`
- `emb_pca_0`, `emb_pca_1`, ..., up to `emb_pca_7`

### 5.10 Composite ground-truth table

Produced by:

- `build_ground_truth()`

Exact columns:

- `zone_id`
- `time_key`
- `y_survival`
- `y_review_quality`
- `y_license_velocity`
- `y_inspection`
- `y_composite`
- `missingness_fraction`
- `label_quality`

Why it exists:

- gives the learned scorer and ranker a target that combines multiple
  merchant-relevant outcomes instead of a single brittle proxy

### 5.11 Learned-model training matrix

Expected by:

- `src/models/train_scoring.py`

Required contract:

- a parquet file named `data/processed/feature_matrix.parquet`
- must contain a `target` column

Columns used by training:

- all columns except `target`
- then drop `zone_id`
- then drop `time_key` or `year` when present

This means the exact learned-model schema is:

- one identifier column family
- one time column family
- one target column
- any number of numeric and encoded feature columns

The current intended feature families are:

- macro zone context
- survival covariates
- NLP demand features
- competition and access features
- cost-pressure features

## 6. Phase-by-Phase Column Usage

### 6.1 Phase: source audit

Dataset:

- dataset audit inventory

Columns:

- `name`
- `owner`
- `spatial_unit`
- `time_grain`
- `earliest_year`
- `status`
- `notes`

### 6.2 Phase: neighborhood feature engineering

Raw source columns currently used in the implemented builder:

- from `licenses`:
  - `event_date`
  - `license_status`
  - `nta_id`
- from `pluto`:
  - `nta_id`
  - `assessed_value`
  - `commercial_sqft`
- from `yelp`:
  - `review_date`
  - `review_text`
  - `zone_id` or `nta_id`
- from `reddit`:
  - schema exists, but conversion to `social_buzz` is still a placeholder
- from `acs`:
  - `year`
  - `nta_id`
  - `median_income`
  - `population`
  - `rent_burden`
- from `inspections`:
  - `inspection_date`
  - `nta_id`
  - `grade`
  - `restaurant_id`

Output columns:

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

### 6.3 Phase: neighborhood phase discovery

Model consumes:

- all numeric columns from the zone-year matrix

Current practical numeric inputs:

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

Output:

- `trajectory_cluster`

### 6.4 Phase: survival modeling

Model consumes the restaurant history table.

Exact table columns:

- `restaurant_id`
- `zone_id`
- `cuisine_type`
- `duration_days`
- `event_observed`
- `inspection_grade_numeric`
- `rent_pressure`
- `competition_score`
- `transit_access`

Current numeric training features:

- `inspection_grade_numeric`
- `rent_pressure`
- `competition_score`
- `transit_access`

Outputs:

- `closure_risk`
- `open_days`

### 6.5 Phase: NLP labeling and demand aggregation

Review labeling input columns / payload:

- raw review text strings
- allowed subtype list

Label output columns:

- `review_id`
- `sentiment`
- `concept_subtype`
- `confidence`
- `rationale`

Aggregated zone-time output columns:

- `zone_id`
- `time_key`
- `healthy_review_share`
- `subtype_gap`
- `dominant_subtype`

Optional NLP enrichment columns:

- `frac_positive`
- `frac_neutral`
- `frac_negative`
- `topic_*_share`
- `embedding_diversity`
- `emb_pca_*`

### 6.6 Phase: final scoring and ranking

Transparent score component fields:

- `healthy_gap_score`
- `subtype_gap_score`
- `demand_signal_score`
- `review_demand_score`
- `merchant_viability_score`
- `license_velocity_score`
- `competition_penalty`
- `rent_pressure_penalty`
- `transit_access_score`
- `income_alignment_score`

Current API heuristic feature fields built per zone:

- `quick_lunch_demand`
- `subtype_gap`
- `survival_score`
- `rent_pressure`
- `competition_score`
- `healthy_review_share`
- `license_velocity`
- `transit_access`
- `income_alignment`
- `healthy_supply_ratio`
- `healthy_gap_score`

Learned-model training contract:

- any tabular feature columns
- mandatory `target`
- identifiers dropped before training

API response columns:

- `zone_id`
- `zone_name`
- `concept_subtype`
- `opportunity_score`
- `confidence_bucket`
- `healthy_gap_summary`
- `positives`
- `risks`
- `freshness_note`
- `feature_contributions`
- `survival_risk`
- `model_version`
- `scoring_path`
- `label_quality`
