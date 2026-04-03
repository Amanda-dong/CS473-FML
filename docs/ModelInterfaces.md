# Model Interfaces and Rationale

Updated: April 3, 2026

This document is the implementation-level companion to `docs/Design.md`.
It answers three questions for every model layer in the project:

1. What exact table or payload goes into the model?
2. What exact predictions, labels, or diagnostics come out?
3. Why is this model a good fit for this project rather than a generic default?

The project is intentionally layered. We do not have one single "restaurant success"
model. We have separate models for neighborhood regime discovery, survival,
review weak labeling, and final ranking.

## 1. Model Inventory

| Layer | Primary code path | Current role | Main input | Main output |
| :---- | :---- | :---- | :---- | :---- |
| Neighborhood phase discovery | `src/models/trajectory_model.py` | Discover unlabeled neighborhood regimes | zone-year feature matrix | `trajectory_cluster` labels + diagnostics |
| Survival baseline | `src/models/survival_model.py` | Estimate merchant survivability under right-censoring | restaurant history table | `closure_risk`, `open_days`, calibration tables |
| NLP weak labeling | `src/nlp/gemini_labels.py` | Create silver labels from review text | review text batch + subtype list | `GeminiReviewLabel` records |
| NLP topic / embedding enrichment | `src/nlp/embeddings.py`, `src/nlp/topic_model.py` | Optional exploratory review features | review texts / embeddings | topic shares, embedding diversity, PCA summaries |
| Review aggregate builder | `src/nlp/review_aggregates.py` | Convert review labels into zone-time features | labeled review table | `healthy_review_share`, `subtype_gap`, etc. |
| Transparent opening score | `src/models/cmf_score.py` | Interpretable product baseline | `ScoreComponents` feature bundle | scalar opening score |
| Learned scoring model | `src/models/cmf_score.py` | Regression-style zone scoring | `feature_matrix.parquet` | predicted score, intervals, SHAP values |
| Learned ranker | `src/models/ranking_model.py` | Top-k ordering of candidate zones | same feature matrix + query groups | ranking scores |

## 2. Neighborhood Phase Discovery

### Why this model exists

The project does not have a trustworthy ground-truth label such as
"this neighborhood is in phase 2 gentrification." That makes unsupervised
clustering the right first step. The clustering layer is used to describe
macro neighborhood context, not to make the final merchant recommendation by
itself.

### Algorithms used

- `KMeans`
- `GaussianMixture`

### Why these algorithms were chosen

- `KMeans` is CPU-cheap, stable, and easy to inspect through cluster centroids.
- `GaussianMixture` is a useful companion because neighborhood change is not
  always cleanly partitioned. GMM gives softer boundaries and BIC/AIC diagnostics.
- Both work well on small-to-medium tabular panels, which matches a class project
  built around zone-year features rather than millions of observations.

### Exact input contract

The intended input table is the output of `build_zone_year_matrix()` in
`src/features/feature_matrix.py`.

Identifiers:

- `zone_id`
- `time_key`

Current implemented numeric feature columns that may be present:

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

Important implementation note:

- `TrajectoryClusteringModel` currently selects **all numeric columns**
  from the frame it is given.
- If `time_key` remains numeric in the training frame, it is included unless
  the caller drops it first.
- In the live `/predict/trajectory` API fallback path, the model is not yet
  fed the zone-year matrix above. It clusters a synthetic/demo feature frame
  built by `_build_features()` in `src/api/routers/recommendations.py`.

The current API fallback feature frame contains these numeric columns:

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

### Preprocessing

- Select numeric columns only
- Fill missing numeric values with `0.0`
- Standardize with `StandardScaler`
- Choose `k` manually or auto-select:
  - silhouette score for k-means
  - BIC for GMM

### Exact outputs

Primary outputs from `TrajectoryClusteringModel`:

- `predict(...) -> pd.Series` named `trajectory_cluster`
- `fit_predict(...) -> pd.Series`
- `describe_clusters(...) -> pd.DataFrame` of mean numeric features by cluster
- `sweep_k(...) -> pd.DataFrame` with:
  - `k`
  - `silhouette`
  - `inertia` for k-means
  - `bic`, `aic` for GMM

Stored diagnostics in `diagnostics_` may include:

- `silhouette`
- `bic`
- `aic`
- `inertia`
- `n_clusters`
- `n_samples`
- `stability_ari`

API output from `/predict/trajectory`:

- `concept_subtype`
- `zone_type`
- `trajectory_cluster`

## 3. Survival Modeling

### Why this model exists

Restaurant openings are a time-to-event problem. A zone can have strong unmet
demand but still be a bad recommendation if businesses close quickly there.
Using survival analysis is more defensible than forcing the problem into binary
classification.

### Algorithms used

- `CoxPHFitter` from `lifelines`
- `RandomSurvivalForest` from `sksurv` when available
- heuristic fallback when model fitting is not possible

### Why these algorithms were chosen

- Cox PH is the best default baseline for this project because it is widely
  understood, interpretable, and appropriate for right-censored durations.
- Random Survival Forest is included because restaurant risk can be nonlinear.
  It is a stronger tabular baseline when the library is available.
- The heuristic fallback keeps the product usable in environments where the
  training artifacts or optional dependencies are missing.

### Exact input contract

The main training table is built by `build_real_restaurant_history()` in
`src/models/survival_model.py`.

Exact output columns of that builder:

- `restaurant_id`
- `zone_id`
- `cuisine_type`
- `duration_days`
- `event_observed`
- `inspection_grade_numeric`
- `rent_pressure`
- `competition_score`
- `transit_access`

How those columns are constructed:

- `duration_days`: days between first observed license event and close date or censoring cutoff
- `event_observed`: `1` if the last observed status is a closed-style status, else `0`
- `inspection_grade_numeric`: average mapped inspection grade (`A=1`, `B=2`, `C=3`)
- `rent_pressure`, `competition_score`, `transit_access`: zone-level covariates merged from zone features

Training features actually used by `SurvivalModelBundle`:

- every numeric column in the restaurant history table
- excluding `duration_days`
- excluding `event_observed`

With the current builder, the numeric covariates are normally:

- `inspection_grade_numeric`
- `rent_pressure`
- `competition_score`
- `transit_access`

### Exact outputs

Prediction outputs:

- `predict_risk(...) -> pd.Series` named `closure_risk`
- `predict_median_survival(...) -> pd.Series` named `open_days`

Evaluation outputs:

- `concordance_index(...) -> float`
- `brier_score(...) -> pd.DataFrame` with columns:
  - `time`
  - `brier_score`
  - `n_informative`
- `calibration_data(...) -> pd.DataFrame` with columns:
  - `predicted_survival`
  - `actual_survival`
  - `count`
  - `bin_error`
- `test_proportional_hazards(...) -> dict`

Saved artifact path:

- `data/models/survival_model.joblib`

## 4. Gemini Weak Labeling for Reviews

### Why this model exists

The project needs a healthy-demand signal from review text, but the team does
not want the core milestone to depend on GPU-heavy local fine-tuning. Gemini is
used as an offline annotator, not as the runtime recommendation engine.

### Model used

- `gemini-2.5-flash-lite`

### Why this model was chosen

- low-latency, low-cost labeling relative to heavier hosted models
- structured JSON output works well for batch annotation
- keeps the main ML contribution in tabular modeling and temporal evaluation,
  not in expensive local transformer training

### Exact input contract

Function: `label_reviews_with_gemini(reviews, subtypes, api_key=None)`

Inputs:

- `reviews`: list of raw review-text strings
- `subtypes`: tuple of allowed concept subtype labels
- `api_key`: explicit key or `GEMINI_API_KEY` from the environment

Prompt contract per review batch:

- request `sentiment`
- request `concept_subtype`
- request `confidence`
- request `rationale`

### Exact outputs

Python object output:

- list of `GeminiReviewLabel`

Exact fields on each label:

- `review_id`
- `sentiment`
- `concept_subtype`
- `confidence`
- `rationale`

Cache artifact:

- `data/processed/gemini_labels.parquet`

Cached parquet columns:

- `review_id`
- `sentiment`
- `concept_subtype`
- `confidence`
- `rationale`

## 5. Review Aggregation and Optional NLP Enrichment

### Why this layer exists

The product does not rank individual reviews. It ranks zones. That means the
review pipeline has to collapse review-level labels into zone-time features that
can be joined into the rest of the tabular system.

### 5.1 Aggregate review labels

Function: `aggregate_review_labels(...)`

Required input columns:

- `review_id`
- `sentiment`
- `concept_subtype`
- `confidence`
- `zone_id`
- `time_key`

Core output columns:

- `zone_id`
- `time_key`
- `healthy_review_share`
- `subtype_gap`
- `dominant_subtype`

Optional extra output columns:

- `frac_positive`
- `frac_neutral`
- `frac_negative`
- `topic_0_share`, `topic_1_share`, ..., `topic_{k-1}_share`

Why this aggregation is used:

- `healthy_review_share` gives a compact demand confirmation signal
- `subtype_gap` captures imbalance inside the category rather than treating all
  healthy concepts as interchangeable
- `dominant_subtype` helps explain what the local market is already signaling

### 5.2 Embeddings

Function: `embed_reviews(texts, model_name="all-MiniLM-L6-v2")`

Input:

- list of review text strings

Output:

- `numpy.ndarray` of shape `(N, 384)` when the sentence-transformer path is available
- fallback is TF-IDF + SVD projected and padded to width `384`

Why used:

- compact semantic representation of review language
- CPU-friendly fallback available
- works well for topic discovery and optional enrichment features

### 5.3 Topic discovery

Function: `discover_topics(embeddings, n_topics=8, texts=None)`

Inputs:

- review embeddings
- optional review texts for top-term extraction

Outputs:

- `cluster_labels`
- `topic_terms`
- `centroids`

Why used:

- identifies recurrent review themes without manual topic labels
- lets the team inspect whether "healthy", "speed", "price", or "service"
  patterns are geographically concentrated

### 5.4 Topic distribution per zone

Function: `topic_distribution_per_zone(...)`

Required input:

- `reviews_df` with `zone_id`
- `cluster_labels`

Output columns:

- `zone_id`
- `topic_0_share`, `topic_1_share`, ..., `topic_{k-1}_share`

### 5.5 Zone embedding features

Function: `compute_zone_embedding_features(...)`

Required input:

- `reviews_df` with `zone_id`
- review embeddings
- review cluster labels

Output columns:

- `zone_id`
- `embedding_diversity`
- `topic_0_share`, `topic_1_share`, ..., `topic_{k-1}_share`
- `emb_pca_0`, `emb_pca_1`, ..., up to `emb_pca_7`

Why used:

- `embedding_diversity` is a compact proxy for how broad or fragmented local
  review topics are
- PCA summaries preserve some semantic structure without making the feature
  matrix explode in width

## 6. Transparent Opening Score

### Why this layer exists

The team needs an interpretable baseline that can always produce a ranking,
even before a learned model is trained. This is also the easiest score to
explain in a class demo and in a recommendation card.

### Exact input contract

`ScoreComponents` fields:

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

Helper function `score_zone_for_concept(...)` currently builds those components
from a zone feature dict whose expected keys are:

- `quick_lunch_demand`
- `healthy_review_share`
- `subtype_gap`
- `survival_score`
- `license_velocity`
- `competition_score`
- `rent_pressure`
- `healthy_supply_ratio`
- `transit_access`
- `income_alignment`
- `healthy_gap_score`

### Exact outputs

- `compute_opening_score(...) -> float`

Why this scoring rule is used:

- transparent enough for a merchant-facing explanation
- easy to tune by feature family
- strong baseline for comparison against the learned scorer and ranker

## 7. Learned Scoring Model

### Why this model exists

Once the team has a composite zone outcome, a learned tabular model can capture
nonlinear interactions that a hand-weighted score will miss.

### Algorithm used

- `xgboost.XGBRegressor`

### Why this algorithm was chosen

- tabular data is the center of this project
- handles nonlinearities and mixed feature scales well
- performs strongly on small-to-medium structured datasets
- integrates well with SHAP for explanation

### Exact input contract

Training script: `src/models/train_scoring.py`

Expected file:

- `data/processed/feature_matrix.parquet`

Mandatory column:

- `target`

Optional identifier columns dropped before training:

- `zone_id`
- `time_key` or `year`

All remaining columns are used as model features.

In other words, the learned scorer is schema-flexible, but the intended current
feature families are:

- macro zone context:
  - `license_velocity`
  - `net_opens`
  - `net_closes`
  - `population`
  - `median_income`
  - `rent_burden`
  - `rent_pressure`
  - `mean_assessed_value`
  - `inspection_grade_avg`
  - `restaurant_count`
- demand / NLP:
  - `healthy_review_share`
  - `social_buzz`
  - `subtype_gap`
  - `dominant_subtype` after encoding if used
  - topic-share and embedding features if added
- recommendation-specific covariates when available:
  - `competition_score`
  - `transit_access`
  - `income_alignment`
  - `healthy_gap_score`
  - `survival_score`

### Exact outputs

Prediction outputs:

- `predict(X) -> np.ndarray`

Uncertainty outputs:

- `predict_with_uncertainty(X) -> (mean_pred, ci_lower, ci_upper)`

Explainability outputs:

- `explain(X) -> pd.DataFrame` with one SHAP column per feature

Saved artifact:

- `data/models/scoring_model.joblib`

## 8. Learned Ranker

### Why this model exists

The end product is a ranked shortlist, not just a regression score. A ranking
objective is therefore a natural stretch goal after the composite target exists.

### Algorithm used

- `xgboost.XGBRanker` with LambdaMART-style objective

### Why this algorithm was chosen

- directly optimizes ordering quality rather than generic squared error
- matches the product requirement of top-k zone recommendation
- stays within the same CPU-friendly tabular-model family as the learned scorer

### Exact input contract

Inputs to `LearnedRanker.fit(...)`:

- `X`: feature table
- `y`: relevance / target score
- `group`: list giving the number of rows in each ranking query group

Current training script simplification:

- the whole train split is treated as one group via `[len(X_train)]`

### Exact outputs

- `predict(X) -> np.ndarray` ranking scores

Saved artifact:

- `data/models/ranking_model.joblib`

## 9. Runtime Recommendation Outputs

The final product layer wraps the models above into `ZoneRecommendation`
records. The API response schema in `src/schemas/results.py` contains:

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

These fields matter because the project is not only trying to predict well. It
is trying to produce a recommendation a merchant can inspect and defend.
