# NYC Healthy-Food Restaurant White-Space Finder
## CS473 Final Presentation — Spring 2026

---

## Slide 1: Title

**NYC Healthy-Food Restaurant White-Space Finder**

Subtype-aware micro-zone scoring for independent restaurant operators.

- Team: Catherine, Harsh, Tony, Siqi, Amanda
- Course: CS473 — Spring 2026
- Date: April 2026

---

## Slide 2: The Problem

- Independent restaurant operators face a structural information asymmetry: chains commission bespoke site-selection analytics; independents guess or rely on intuition
- NYC has 195 distinct Neighborhood Tabulation Areas — which ones have latent healthy-food demand with no adequate supply?
- Three compounding difficulties:
  - Survivorship bias in public data (only active licenses are visible)
  - Platform coverage gaps (Yelp over-indexes upscale; DCA licenses cover the full universe)
  - Micro-zone vs. neighborhood granularity — a borough-level signal masks opportunity at the block-cluster level

> **Speaker note:** This is not a generic restaurant recommender. It is specifically about finding white space for healthy fast-casual concepts — the decision a first-time independent operator actually faces.

---

## Slide 3: Our Approach

Pipeline overview:

```
Raw NYC Open Data  -->  ETL (10 sources)  -->  Feature Matrix
Feature Matrix  -->  Survival Model + CMF Score  -->  Zone Rankings
Zone Rankings  -->  FastAPI  -->  Streamlit UI  -->  Merchant Decision
```

Key architectural choices:

- Official DCA license data as the restaurant universe backbone, not Yelp
- Survival modeling, not binary classification — preserves right-censored observations
- Subtype gap score, not just healthy vs. unhealthy — the healthy-food market is not homogeneous

---

## Slide 4: Data Architecture

**Tier 1 — Official sources (ground truth):**

- DCA Business Licenses: full restaurant universe, open/close dates, license status
- DOHMH Inspection grades: A/B/C/P/Z history per establishment
- DOB Building Permits: construction pressure as rent-increase proxy
- Census ACS: income distribution, household density per NTA
- NTA Boundaries: spatial join layer for all zone-level aggregations

**Tier 2 — Enrichment sources:**

- Yelp Fusion: review text, rating distributions, subtype labels
- Citi Bike trip data: mobility and transit catchment signal
- Inside Airbnb: short-term rental density as gentrification indicator
- NYC 311 Complaints: neighborhood stress / code violation density

**Key design decision:** DCA license data, not Yelp, defines the restaurant universe. Yelp coverage skews toward mid-market and above; DCA captures every licensed food establishment regardless of digital footprint.

**ETL output:** Multiple processed Parquet tables joined into a **micro-zone × year** feature matrix (`data/processed/feature_matrix.parquet`; on-disk baseline **726 rows × 49 columns**), with `nta_id` used as a join key where sources are NTA-native.

---

## Slide 5: Neighborhood Phase Discovery

- K-Means clustering over time-windowed feature vectors per NTA
  - Input features: license velocity delta, review growth rate, demographic drift (ACS year-over-year), permit intensity
  - Window: rolling 3-year slices from 2015 to 2024
- Output: four user-facing trajectory labels (clustering is unsupervised; labels are mapped for the API)
  - **Emerging** — rising license velocity, demographic shift, permit activity
  - **Fast-growing** — review growth and income influx ahead of supply change (API string; narrative slides may still say “gentrifying-like” dynamics)
  - **Stable** — low variance across all signals
  - **Declining** — license attrition, flat or falling demand signals
- Validated qualitatively against NYU Furman Center neighborhood change narratives

> **Speaker note:** This phase label provides macro context that conditions downstream estimates. Knowing a zone is "emerging" vs. "declining" meaningfully shifts the prior on survival and demand signals. We treat it as a categorical covariate, not a replacement for the other features.

---

## Slide 6: Restaurant Survival Modeling

**Why survival analysis, not classification:**

- Standard classification treats a still-open restaurant as a positive label — this discards timing information
- Survival analysis handles **right-censoring**: a restaurant still open at the data cutoff has an **observed** tenure so far but a **future** closure time that is unknown — that row is censored, not a “failed” label

**Model:** Cox Proportional Hazards as primary baseline; Random Survival Forest and heuristic paths exist in code when libraries or convergence fail.

- Covariates (high level): zone-level rent, competition, inspection, transit; plus **license-history** features from the event sequence (`n_renewals`, `mean_renewal_interval_days`, `n_inactive_events`, `days_since_last_event`) — see `docs/EvaluationResults.md` §5.3
- Event definition: DCA license last status in a closed set (e.g. inactive / expired / surrendered / revoked — see `build_real_restaurant_history()` in `src/models/survival_model.py`)

**Results (sync with `docs/EvaluationResults.md` before presenting):**

- Cox PH holdout C-index **0.5544**; 5-fold CV **0.5992 ± 0.0043**; bootstrap CI **[0.591, 0.608]** (same table in Evaluation Results)

**Key finding:** License-history and inspection signals dominate discriminative power in the current bundle — treat “inspection-only” claims as outdated if you cite older slides.

---

## Slide 7: CMF Opportunity Score

Weighted sum of 10 normalized signals, each scaled to [0, 1]:

| Signal | Weight | Rationale |
|---|---|---|
| Demand signal | 0.20 | Review velocity + NLP healthy-demand share |
| Merchant viability | 0.18 | Survival model output for the zone-concept pair |
| Subtype gap | 0.16 | Intra-category variance across healthy subtypes |
| Healthy gap | 0.12 | Overall healthy supply deficit vs. demand |
| License velocity | 0.10 | Recent net new licenses (market momentum) |
| Review demand (NLP) | 0.08 | Review-derived healthy-demand share (`review_demand_score` in code) |
| Transit access | 0.07 | Trip / mobility proxies (e.g. Citi Bike counts normalized in the feature dict) |
| Competition penalty | 0.08 | Established competitor density (negative) |
| Rent penalty | 0.04 | Permit-derived rent pressure (negative) |
| Income alignment | 0.05 | ACS income match to concept price tier |

**Core design decision on subtype gap:** Standard deviation of per-subtype proportions across healthy concepts. A zone saturated with Mediterranean options but with zero healthy Indian supply scores high on subtype gap even if its aggregate healthy-food supply looks adequate. This is the thesis: the healthy-food market is not homogeneous.

> **Speaker note:** The subtype gap signal is what differentiates this system from a generic "find an underserved neighborhood" tool. It forces the model to reason about the internal structure of the healthy-food category.

---

## Slide 8: NLP Pipeline

**Challenge:** Healthy-food demand signal must come from review text; no labeled dataset exists for NYC restaurant subtypes at this granularity.

**Solution:**

- Gemini Flash-Lite as offline batch annotator
- Reviews processed in batches; outputs cached as Parquet (silver labels, not ground truth)
- 7 concept subtypes assigned per review:
  - `healthy_indian` — `mediterranean_grain_bowl` — `vegan_vegetarian`
  - `salad_bowl` — `quick_grab_and_go` — `unhealthy_dominant` — `neutral`

**Current status:**

- Keyword-regex fallback is active in production (covers explicit mentions of subtype terms)
- Gemini annotation pass is in progress; will replace regex once quality threshold is validated
- Label quality estimate pending held-out human review sample

**Three derived features used downstream (zone-year matrix naming):**

- `healthy_food_share` — healthy-food demand share aggregated to the zone-year row (not `healthy_review_share` in the current parquet)
- `subtype_gap` — spread across per-subtype proportions / taxonomy gap signal
- `dominant_subtype` — modal subtype label for a zone (used for concept-match scoring)

---

## Slide 9: Evaluation — Temporal Backtest

**Validation design:** Walk-forward expanding window. No random train/test splits — temporal leakage would inflate every metric.

- Train on years 1..t, evaluate on year t+1
- Task: rank a held-out set of zones; measure shortlist quality for a top-5 recommendation

**Primary metric:** NDCG@5 (normalized discounted cumulative gain at rank 5) — appropriate for a shortlist recommendation task where rank order matters.

| Fold year | NDCG@5 | Precision@5 | MAP |
|-----------|--------|-------------|-----|
| 2020 | 0.9765 | 0.800 | 0.877 |
| 2022 | 0.9667 | 0.600 | 0.628 |
| 2023 | 0.9617 | 0.600 | 0.659 |

*(Values copied from `docs/EvaluationResults.md` §5.1 walk-forward table; re-run `src/validation/run_evaluation.py` if you need fresher numbers.)*

**Trend interpretation:**

- Metrics stay high across folds (short catalog + strong temporal features). Written interpretation and COVID narrative in `EvaluationResults.md` should be read together with the table — do not invent a “steady climb” story if the printed fold rows disagree.

---

## Slide 10: Evaluation — Feature Ablation

Which signals actually drive ranking quality? Leave-one-group-out ablation (values from `docs/EvaluationResults.md` §5.2 — full model NDCG@5 = **0.9706**):

| Group removed | NDCG@5 after ablation | Drop from full |
|---------------|----------------------|----------------|
| demand | 0.801 | 0.170 |
| survival | 0.852 | 0.119 |
| nlp | 0.897 | 0.074 |
| rent_cost | 0.926 | 0.045 |
| competition | 0.942 | 0.029 |

**Take-aways:**

- **Demand** group removal hurts most — consistent with the evaluation write-up
- **Survival** is second — merchant viability signal matters for ranking
- NLP, rent/cost, and competition matter but with smaller marginal drops on this table

---

## Slide 11: Demo — Product Walkthrough

Streamlit shell: **two in-app tabs** on `frontend/app.py` — **Top Picks** and **Data Sources**. **Methodology** is a **separate multipage entry** (`frontend/pages/1_Methodology.py`), not a third tab on the main screen.

**Merchant workflow:**

1. Select concept (e.g., "Healthy Indian") + price tier + risk tolerance slider
2. System scores all 137 modeled micro-zones and returns the top 5 ranked by opportunity score
3. Each result card displays:
   - Zone type badge (e.g. campus walkshed, lunch corridor, transit catchment, business district)
   - Opportunity score (0–100 in UI — scaled from the 0–1 backend score)
   - Survival risk percentage
   - Confidence interval
   - Trajectory cluster label (emerging / fast-growing / stable / declining)
   - Risk flags (e.g., "high competition density", "rent pressure above median")
   - Positive drivers (e.g., "strong healthy demand signal", "subtype gap: no Indian supply")
   - Score breakdown by signal group
4. Side-by-side concept comparison (e.g., Mediterranean Bowls vs. Healthy Indian for the same zone set)
5. Export shortlist as CSV for offline use

> **Speaker note:** The UI is intentionally shortlist-first, not map-first. A map encourages browsing; a ranked shortlist encourages a decision. The target user has limited time and needs a defensible short list to walk into a lease negotiation.

---

## Slide 12: Key Findings

**Top white-space zones by concept (2024 model output):**

- Healthy Indian: Fordham / Bronx Campus Belt, Crown Heights, Flatbush
- Mediterranean Bowls: Mott Haven (low supply, rising demand signal), Sunset Park (diverse existing mix, gap for grain-bowl formats)
- Salad Bowls: Co-op City (underserved business district, captive weekday lunch demand)

**Cross-concept patterns:**

- Campus walk-sheds consistently show lower survival risk than CBD business districts — stable demand from a enrolled population vs. economic-cycle-sensitive office foot traffic
- Transit-adjacent zones score highest on demand signals but also carry the highest competition penalty — the opportunity signal is real but the window may already be closing
- Zones labeled "emerging" by the phase model outperform "stable" zones in 2-year survival for new entrants — consistent with first-mover dynamics in fast-changing corridors

---

## Slide 13: Limitations and Future Work

**Current limitations:**

- ACS income data: synthetic fallback used for several NTAs where ACS suppression thresholds apply; real ACS microdata access would improve income-alignment precision
- Zone catalog: 137 modeled micro-zones; expanding to all 195 NTAs or H3 hexagon resolution requires additional feature engineering and compute
- Mobility proxy: Citi Bike trip counts are a noisy mobility signal; foot-traffic panels (Placer.ai, Safegraph) would be materially stronger but are cost-prohibitive for a course project
- NLP label quality: Gemini annotation quality estimate is pending human review; keyword-regex is a known lower bound

**Planned extensions:**

- Demographic shift forecasting: use ACS trajectory to project 3-year income and household-composition change per zone
- Real-time demand signals: connect to live Yelp Fusion + DCA license feeds rather than static annual snapshots
- Multi-city extension: the pipeline is city-agnostic given an equivalent license registry; Chicago and LA are natural next targets

---

## Slide 14: Conclusion

**What we built:**

- A rigorous, data-driven healthy-food white-space recommender for NYC micro-zones, end-to-end from raw open data to a live Streamlit application

**Four technical contributions:**

1. Subtype-aware gap scoring — the first signal that quantifies intra-category variance within the healthy-food market
2. Survival-modeled risk — correctly handles right-censoring; provides calibrated risk estimates, not classification labels
3. Temporal walk-forward validation — no leakage; results are comparable to real deployment conditions
4. Decision-ready UI — ranked shortlist with score decomposition and export, not a map for browsing

**System status:** End-to-end pipeline is operational. Models trained. Streamlit app deployed locally. NLP annotation pass in progress.

Open for questions.
