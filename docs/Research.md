# Research & Rationale

Updated: April 30, 2026

## Executive summary

The project positions **official NYC and federal public data** as the backbone,
uses **third-party text and platform data** only as audited enrichment, and
stacks **unsupervised regime discovery**, **survival modeling**, **offline
LLM labeling**, and **tabular scoring / ranking** into one time-aware pipeline.
The frontend is a **shortlist-and-evidence** flow for healthy-food white space,
not a map-first dashboard.

The repository has moved from research notes to a **shipped implementation**:
see `docs/Proposal.md` for product framing, `docs/Design.md` for layout and
ownership, and `docs/ModelInterfaces.md` for exact model and API contracts.

## Verified findings

### Data-source reality

- The U.S. Census Bureau continues to provide ACS 5-year data through the
  official API, which is suitable for stable demographic and housing features.
- Inside Airbnb limits free historical availability and discourages repeated
  scraping; treat it as a constrained source, not a full longitudinal backbone.
- Reddit Data API access is governed by formal terms; Reddit remains optional
  and replaceable (this build uses **NYC 311** as the primary coarse social
  signal where Reddit is thin).
- **`pytrends` (Google Trends)** is an unofficial, archived wrapper and is
  **not** used in this project.

### Modeling reality

- scikit-learn guidance for `TimeSeriesSplit` stresses that shuffling is
  inappropriate for time-ordered data; headline evaluation here uses **blocked /
  rolling temporal** splits, not random train/test.
- XGBoost documents both AFT survival extensions and **learning-to-rank**
  (LambdaMART-style) objectives; this repo uses Cox PH + optional RSF for
  survival and `XGBRanker` for ranking when trained.
- Hosted LLMs are used as **offline annotators** for review text; runtime
  recommendation stays on tabular models plus transparent CMF scoring.

### Tooling reality

- **`uv`** is a mature default for fast env setup and `uv run` workflows; the
  repo also documents conda + venv + pip.
- Gemini labeling in code defaults to **`gemini-2.5-flash-lite`**
  (`src/nlp/gemini_labels.py`). Prefer the [official Gemini model
  catalog](https://ai.google.dev/models/gemini) when rotating model strings.

## Design choices (implemented)

1. **Shortlist-first UX** — Users get top zones, evidence, risk, and confidence
   before map exploration (`frontend/`).
2. **Layered ML** — Trajectory clustering (k-means / GMM), survival (Cox +
   optional RSF), Gemini silver labels → zone aggregates, interpretable CMF
   score, XGBoost scorer, LambdaMART ranker.
3. **LLMs for labeling, not live scoring** — Lower inference cost, cleaner
   offline evaluation, reproducible feature columns in
   `data/processed/feature_matrix.parquet`.
4. **Official city data first** — Licenses, permits, inspections, PLUTO, ACS,
   311, boundaries; Yelp and Reddit as enrichment with coverage discipline.
5. **Micro-zones** — Recommendation unit is walk sheds / corridors /
   catchments (see `src/features/microzones.py`), not only whole NTAs.
6. **Healthy subtype gaps** — Signals distinguish Mediterranean saturation vs.
   healthy Indian / South Asian white space (`src/utils/taxonomy.py`,
   `src/features/healthy_gap.py`).

Recommended offline label pipeline (aligned with `src/nlp/gemini_labels.py`):

1. Batch-call Gemini for sentiment, subtype, confidence, short rationale.
2. Filter to high-confidence rows after spot checks.
3. Maintain a 200–300 review **gold** set for agreement metrics.
4. Aggregate to `zone_id` × `time_key` features in `review_aggregates.py`.
5. Optional later: distill labels into a CPU classifier if API cost dominates.

## Dependency notes

Pins are centralized in `requirements.txt` (updated 2026-04-25). The team
**intentionally upgraded** several previously “high-risk” majors (e.g. pandas
2.3.x, scikit-learn 1.8.x, xgboost 3.2.x, transformers 5.6.x) after
compatibility checks with lifelines and the test suite (~606 tests).

| Area | Notes |
| :---- | :---- |
| Core stack | pandas, numpy, geopandas, scipy, pyarrow — pinned for reproducibility |
| ML | scikit-learn, xgboost, lifelines; optional **scikit-survival** for RSF when installed |
| NLP / LLM | `google-genai`, `transformers`, `sentence-transformers` (embeddings path) |
| API / UI | FastAPI, uvicorn, pydantic; Streamlit |
| **Removed** | `pytrends`; Google Trends is out of scope |

Before bumping pins again, run the full pytest suite and regression checks on
`data/processed/feature_matrix.parquet` consumers.

## Milestones (delivery arc)

The following phases match the sprint plan in `docs/Sprints.md`; all are
**complete** in the current repo.

| Phase | Focus |
| :---- | :---- |
| 1 | Temporal audit, Yelp NYC coverage discipline, 311 / boundaries ETL, frozen feature-matrix contract |
| 2 | Zone-year matrix (726 × 49), micro-zones (137), trajectory clustering, supply-gap features |
| 3 | Survival training, Gemini labeling + aggregates, FastAPI + Streamlit integration |
| 4 | Temporal backtests, ablations, causal checks, ranking model, report and evaluation artifacts |

Detailed metrics: `docs/EvaluationResults.md`, `docs/CausalMLEvaluationReport.md`.

## Sources

- U.S. Census Bureau ACS 5-Year API: https://www.census.gov/data/developers/data-sets/acs-5year.2016.html
- Inside Airbnb Data Policies: https://beta.insideairbnb.com/data-policies/
- Reddit Data API Terms: https://redditinc.com/policies/data-api-terms
- Archived `pytrends` (not a project dependency): https://github.com/GeneralMills/pytrends
- scikit-learn `TimeSeriesSplit`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- XGBoost AFT Survival Tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
- XGBoost Learning to Rank Tutorial: https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
- Hugging Face sequence classification guide: https://huggingface.co/docs/transformers/tasks/sequence_classification
- uv documentation: https://docs.astral.sh/uv/
- Gemini models: https://ai.google.dev/models/gemini
- Gemini API documentation: https://ai.google.dev/gemini-api/docs/gemini-3
- Gemini deprecations: https://ai.google.dev/gemini-api/docs/deprecations
