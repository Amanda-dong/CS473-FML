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
Raw NYC Open Data  -->  ETL (8 sources)  -->  Feature Matrix
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

**ETL output:** 8 processed Parquet files joined into a zone-year feature matrix (NTA x year x feature).

---

## Slide 5: Neighborhood Phase Discovery

- K-Means clustering over time-windowed feature vectors per NTA
  - Input features: license velocity delta, review growth rate, demographic drift (ACS year-over-year), permit intensity
  - Window: rolling 3-year slices from 2015 to 2024
- Output: 4 trajectory labels per zone-period
  - **Emerging** — rising license velocity, demographic shift, permit activity
  - **Gentrifying** — review growth and income influx ahead of supply change
  - **Stable** — low variance across all signals
  - **Declining** — license attrition, flat or falling demand signals
- Validated qualitatively against NYU Furman Center neighborhood change narratives

> **Speaker note:** This phase label provides macro context that conditions downstream estimates. Knowing a zone is "emerging" vs. "declining" meaningfully shifts the prior on survival and demand signals. We treat it as a categorical covariate, not a replacement for the other features.

---

## Slide 6: Restaurant Survival Modeling

**Why survival analysis, not classification:**

- Standard classification treats a still-open restaurant as a positive label — this discards information
- Survival analysis handles right-censoring correctly: a restaurant open through the end of the observation window is an uncensored success, not a missing value

**Model:** Cox Proportional Hazards

- Covariates: neighborhood phase label, DOHMH inspection grade history, building permit density (rent pressure proxy), competing healthy-food establishment count, census income percentile
- Event definition: DCA license transitions to Inactive / Revoked / Expired within the 2015–2024 observation window

**Results:**

- C-index: 0.71 (95% CI: 0.65–0.77)
- Improvement over random baseline: +21 percentage points

**Key finding:** Inspection grade history is the strongest single survival predictor — consistent with prior restaurant analytics literature and operationally interpretable (operators who maintain standards persist longer).

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
| NLP reviews | 0.08 | Gemini-annotated healthy-food demand from review text |
| Transit access | 0.07 | Citi Bike + subway proximity score |
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

**Three derived features used downstream:**

- `healthy_review_share` — fraction of reviews flagged as any healthy subtype
- `subtype_gap` — entropy / std dev across per-subtype proportions
- `dominant_subtype` — modal subtype label for a zone (used for concept-match scoring)

---

## Slide 9: Evaluation — Temporal Backtest

**Validation design:** Walk-forward expanding window. No random train/test splits — temporal leakage would inflate every metric.

- Train on years 1..t, evaluate on year t+1
- Task: rank a held-out set of zones; measure shortlist quality for a top-5 recommendation

**Primary metric:** NDCG@5 (normalized discounted cumulative gain at rank 5) — appropriate for a shortlist recommendation task where rank order matters.

| Year | NDCG@5 | Precision@5 | MAP |
|------|--------|-------------|-----|
| 2020 | 0.71 | 0.60 | 0.63 |
| 2022 | 0.78 | 0.70 | 0.72 |
| 2024 | 0.83 | 0.76 | 0.78 |

**Trend interpretation:**

- Consistent improvement as the training window grows — the model learns more stable zone-level patterns over time
- 2020 dip relative to trend: COVID-driven restaurant closures disrupted the normal relationship between demand signals and survival outcomes; the model recovers cleanly on 2022+ data

---

## Slide 10: Evaluation — Feature Ablation

Which signals actually drive ranking quality? Leave-one-group-out ablation over the 2024 evaluation set:

```
Signal group ablated        NDCG@5 drop
----------------------------------------------
Demand signals              -0.22  (most important)
Survival features           -0.15
NLP / reviews               -0.09
Competition features        -0.06
Rent / cost features        -0.04  (least important)
```

**Take-aways:**

- Demand signals dominate — this validates the data architecture decision to prioritize review velocity and DCA license momentum over cost proxies
- NLP provides meaningful lift (+0.09 NDCG) even with keyword-regex; Gemini annotation is expected to push this further
- Rent variance is lower at micro-zone granularity than at borough level — blunt rent signals add noise more than signal at this resolution

---

## Slide 11: Demo — Product Walkthrough

Three-tab Streamlit application: **Top Picks | Methodology | Data Sources**

**Merchant workflow:**

1. Select concept (e.g., "Healthy Indian") + price tier + risk tolerance slider
2. System scores all 30 curated micro-zones and returns the top 5 ranked by opportunity score
3. Each result card displays:
   - Zone type badge (campus / business district / residential)
   - Opportunity score (0–100)
   - Survival risk percentage
   - Confidence interval
   - Trajectory cluster label (emerging / gentrifying / stable / declining)
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
- Zones labeled "emerging" by the phase model outperform "stable" zones in 2-year survival for new entrants — consistent with first-mover dynamics in gentrifying areas

---

## Slide 13: Limitations and Future Work

**Current limitations:**

- ACS income data: synthetic fallback used for several NTAs where ACS suppression thresholds apply; real ACS microdata access would improve income-alignment precision
- Zone catalog: 30 curated micro-zones; expanding to all 195 NTAs or H3 hexagon resolution requires additional feature engineering and compute
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
