"""Methodology page with real explanatory content."""

from __future__ import annotations

import streamlit as st


def render_methodology_page() -> None:
    """Render the methods explainer with expandable sections."""

    st.subheader("Methodology")
    st.write(
        "The NYC Restaurant Intelligence Platform combines geospatial, transactional, "
        "and text signals to surface healthy-food white-space recommendations."
    )

    with st.expander("1. Data Sources"):
        st.markdown(
            """
| # | Source | Description |
|---|--------|-------------|
| 1 | **NYC DCA License Events** | Restaurant license open/close events by NTA and year |
| 2 | **NYC PLUTO** | Parcel-level assessed value and commercial square footage |
| 3 | **Yelp Fusion API** | Business listings, categories, review counts, ratings |
| 4 | **Google Places API** | Supplementary location data and category signals |
| 5 | **NYC 311 Complaints** | Noise/food-safety complaints as negative demand proxies |
| 6 | **NYC Health Inspections** | Inspection grades and violation counts per restaurant |
| 7 | **MTA Turnstile Data** | Station-level ridership as foot-traffic proxy |
| 8 | **Census ACS 5-Year** | Block-group income, education, and household composition |
| 9 | **NYC Neighborhood Tabulation Areas** | Administrative zone boundaries for spatial joins |
| 10 | **Yelp Review Text** | Full review text used for NLP labeling pipeline |
| 11 | **NYC Open Data – DOHMH** | Health department restaurant inspection result details |
| 12 | **Google Trends (local)** | Search-interest proxy for healthy-food demand trends |
"""
        )

    with st.expander("2. Neighborhood Phase Discovery"):
        st.markdown(
            """
**Algorithm:** K-Means (primary) and Gaussian Mixture Models (GMM, secondary)

Each neighborhood-time observation is represented as a feature vector covering:
- License velocity (net opens minus closes per year)
- Rent pressure (normalized assessed value trajectory)
- Healthy review share (from NLP pipeline)
- Competition density (direct competitors per 0.5 km radius)

K-Means partitions zones into *k* = 3–5 clusters representing macro regimes:
`emerging`, `saturated`, `stable`, and `declining`.

GMM adds soft-assignment probabilities useful for borderline zones.
Both models use `StandardScaler` normalization. Hyperparameters are
selected via silhouette score on a held-out validation split.
"""
        )

    with st.expander("3. Survival Modeling"):
        st.markdown(
            """
**Algorithm:** Cox Proportional Hazards (lifelines `CoxPHFitter`)

The survival model estimates the probability that a newly opened restaurant
survives beyond a given number of days, conditioned on zone-level covariates.

**Duration variable:** `duration_days` — days from license open to close (or censoring)

**Event variable:** `event_observed` — 1 if the restaurant closed, 0 if still active

**Covariates:** rent pressure, competition score, inspection grade (numeric),
license velocity of the surrounding zone.

The partial-hazard scores are normalized to [0, 1] and used as the
`survival_score` input to the CMF opening score.
"""
        )

    with st.expander("4. NLP Labeling (Gemini Weak Labeling)"):
        st.markdown(
            """
**Model:** Gemini 2.5 Flash Lite via `google-genai` SDK

Review texts are batched in groups of 10 and sent to Gemini with a
structured prompt requesting:
- `sentiment`: positive / neutral / negative
- `concept_subtype`: one of the allowed taxonomy labels
- `confidence`: float in [0, 1]
- `rationale`: brief explanation

Labels with `confidence < 0.7` are discarded. The remaining labels are
aggregated at (zone, year) granularity to produce:
- **healthy_review_share**: fraction of positive reviews
- **dominant_subtype**: most frequently labeled concept subtype
- **subtype_gap**: standard deviation of normalized subtype proportions
  (high variance → unmet demand for underrepresented subtypes)

When no API key is present the pipeline falls back to deterministic
synthetic labels for local development and CI.
"""
        )

    with st.expander("5. Healthy Gap Scoring"):
        st.markdown(
            r"""
**Formula:**

```
healthy_gap_score = max(0,
    quick_lunch_demand × 0.50
  + subtype_gap        × 0.35
  - healthy_supply_ratio × 0.25
)
```

**Opening score (CMF):**

```
opening_score =
    healthy_gap_score       × 0.35
  + subtype_gap_score       × 0.25
  + merchant_viability_score × 0.30
  - competition_penalty      × 0.10
```

**Merchant viability:**

```
merchant_viability = max(0,
    survival_score   × 0.50
  - rent_pressure    × 0.25
  - competition_score × 0.25
)
```

All inputs are normalized to [0, 1] before scoring.
"""
        )

    with st.expander("6. Temporal Validation"):
        st.markdown(
            """
**Strategy:** Blocked time-series splits (no data leakage across years)

The dataset spans 2014–2023. Validation uses rolling-origin blocked splits:
- **Train:** years *t* through *t + k*
- **Validation:** year *t + k + 1*
- **Test:** years 2022–2023 (held out until final evaluation)

This prevents temporal leakage: features computed from future license events
or future rent trajectories never appear in training folds.

Key metrics tracked across folds:
- Spearman rank correlation of predicted vs. realized zone performance
- Precision@5 for top recommended zones
- Calibration of survival probability estimates (Brier score)
"""
        )
