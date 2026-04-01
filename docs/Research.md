# SOTA Research Refresh

Updated: April 1, 2026

## Executive Summary

The original project idea is strong, but the first draft of the docs leaned too hard on fragile data sources and not hard enough on the parts that would make the system genuinely useful. The strongest version of this project uses official NYC data as the backbone, treats third-party APIs as enrichment, keeps the ML stack layered and time-aware, and turns the frontend into a shortlist-and-evidence tool for underserved healthy-food zones rather than a generic map dashboard.

## Verified Findings

### Data-source reality

- The U.S. Census Bureau continues to provide ACS 5-year data through the official API, which is suitable for stable demographic and housing features.
- Inside Airbnb explicitly limits free historical availability and discourages repeated scraping, so it should be treated as a constrained source rather than assumed to be a full longitudinal backbone.
- Reddit Data API access is governed by formal terms and restrictions, so Reddit should remain optional and replaceable.
- `pytrends` is still an unofficial Google Trends wrapper and the upstream repository is archived and read-only, which makes it a poor core dependency for a class project.

### Modeling reality

- Official scikit-learn guidance for `TimeSeriesSplit` states that shuffling is inappropriate for time-ordered data.
- XGBoost provides official tutorials for both AFT survival analysis and learning-to-rank, which makes them strong extensions if the team has time.
- Hugging Face continues to support standard sequence-classification fine-tuning flows, but that should be treated as optional here because the user explicitly wants to minimize training time and GPU use.

### Tooling reality

- `uv` is now a mature Python project and package manager with strong performance and workspace support, so it is the right default for the docs instead of a `pip`-first workflow.
- As of April 1, 2026, official Google Gemini docs clearly list `gemini-2.5-flash-lite`, `gemini-2.5-flash-lite-preview-09-2025`, and `gemini-3-flash-preview`. I could not verify an official model string named `gemini-3.1-flash-lite-preview`.

## Project Recommendations

These are inferences from the findings above and from the repo state.

### 1. Make the product shortlist-first

Do not center the UX on a map and a wall of filters. The best user flow is:

- enter a healthy concept subtype and constraints
- receive the top 5 underserved zones
- inspect evidence, risk, and confidence
- compare scenarios

That is what makes the system frictionless and useful to an operator.

### 2. Keep the ML story layered

The standout version of the project is:

- unsupervised neighborhood regime discovery
- survival modeling on official restaurant-license records
- Gemini-assisted weak labeling for reviews
- direct aggregation of review labels into healthy-demand features
- concept-subtype white-space detection within the healthy category
- optional small ranking layer for final underserved-zone ordering

This is much stronger than a single hand-weighted heuristic score.

### 3. Use LLMs for labeling, not for live inference

For this project, hosted LLMs are best used as high-throughput annotators, not as the runtime scoring engine.

Why:

- lower cost at inference time
- easier reproducibility
- cleaner offline evaluation
- stronger course-project ML contribution

Recommended low-compute label pipeline:

1. Prompt a Gemini Flash or Flash-Lite model for sentiment and short rationale labels.
2. Keep only high-confidence outputs after spot-checking.
3. Build a 200 to 300 example manually labeled gold set.
4. Aggregate labels directly into neighborhood-level signals.
5. Only if needed later, distill into a lightweight CPU-friendly classifier.

### 4. Keep official city data as the backbone

Priority order:

- official NYC operations and regulatory data
- official federal demographic data
- stable public mobility or housing-pressure sources
- third-party review and forum data only as enrichment

This reduces project risk and makes the empirical story easier to defend.

### 5. Use micro-zones, not only neighborhoods

This is an inference from the merchant use case.

If the product is trying to identify a place like a campus-adjacent lunch district with weak healthy-food options, neighborhood-level output is too coarse. The better recommendation unit is a micro-zone such as:

- a campus walk shed
- a transit-centered lunch corridor
- a business-district catchment
- a small grid or H3 cell

Neighborhood context still matters, but the final white-space recommendation should be made at the micro-zone level.

### 6. Model concept gaps inside the healthy category

This is also an inference from the merchant use case.

The system should not stop at “healthy food is under-supplied here.” It should also identify which healthy subtype is under-supplied. A zone with several bowl or Mediterranean concepts may still have a gap for healthy Indian or South Asian fast casual.

That makes the recommendation materially more useful because it answers not only where to open, but also what to open there.

## Dependency Delta

Requirements scanned: 16 declared Python dependencies from `requirements.txt`.

Core package spot-checks on April 1, 2026:

| Package | Current repo pin | Latest observed | Recommendation |
| :---- | :---- | :---- | :---- |
| pandas | 2.2.3 | 3.0.2 | upgrade later, but treat as high-risk |
| numpy | 2.1.3 | 2.4.4 | upgrade later, moderate risk |
| geopandas | 1.0.1 | 1.1.3 | moderate-risk upgrade |
| scikit-learn | 1.6.1 | 1.8.0 | high-risk upgrade |
| xgboost | 2.1.4 | 3.2.0 | high-risk upgrade |
| lifelines | 0.30.0 | 0.30.3 | low-risk upgrade |
| fastapi | 0.115.8 | 0.135.2 | moderate-risk upgrade |
| uvicorn | 0.34.0 | 0.42.0 | low-risk to moderate-risk upgrade |
| streamlit | 1.42.2 | 1.56.0 | moderate-risk upgrade |
| transformers | 4.48.2 | 5.4.0 | high-risk upgrade |
| matplotlib | 3.10.0 | 3.10.8 | low-risk upgrade |
| seaborn | 0.13.2 | 0.13.2 | already current |
| pytest | 8.3.4 | 9.0.2 | low-risk to moderate-risk upgrade |
| python-dotenv | 1.0.1 | 1.2.2 | low-risk upgrade |
| praw | 7.8.1 | 7.8.1 | already current |
| pytrends | 4.9.2 | 4.9.2 | remove instead of upgrade |

High-risk upgrades:

- pandas 3.x
- scikit-learn 1.8
- xgboost 3.x
- transformers 5.x

Enterprise baseline:

- keep the current pins while the project is still scaffold-only
- remove `pytrends`
- switch docs and scripts to `uv`
- add new dependencies only when the related module is actually implemented

## 8-Week Moat Plan

### Phase 1: Lock the backbone

- complete the temporal coverage audit
- audit Yelp NYC coverage
- decide whether Reddit survives or is replaced by 311
- freeze the core feature matrix schema

### Phase 2: Build the ML spine

- ship neighborhood regime discovery
- ship survival baselines
- ship Gemini-assisted label generation and healthy-demand aggregation

### Phase 3: Make the product useful

- turn the score into a shortlist of underserved healthy-food zones
- expose reasons and risks with every result
- add confidence and freshness signals
- support what-if changes to healthy concept subtype and risk tolerance

### Phase 4: Distinguish the project

- if time remains, add a learned ranking layer
- evaluate ranking quality on held-out periods
- compare the learned ranker against the interpretable weighted baseline

## Sources

- U.S. Census Bureau ACS 5-Year API: https://www.census.gov/data/developers/data-sets/acs-5year.2016.html
- Inside Airbnb Data Policies: https://beta.insideairbnb.com/data-policies/
- Reddit Data API Terms: https://redditinc.com/policies/data-api-terms
- `pytrends` repository: https://github.com/GeneralMills/pytrends
- scikit-learn `TimeSeriesSplit`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- XGBoost AFT Survival Tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
- XGBoost Learning to Rank Tutorial: https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
- Hugging Face sequence classification guide: https://huggingface.co/docs/transformers/tasks/sequence_classification
- uv documentation: https://docs.astral.sh/uv/
- Gemini model catalog: https://ai.google.dev/models/gemini
- Gemini 3 developer guide: https://ai.google.dev/gemini-api/docs/gemini-3
- Gemini deprecations: https://ai.google.dev/gemini-api/docs/deprecations
