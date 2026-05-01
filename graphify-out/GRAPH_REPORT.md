# Graph Report - cs473-fml  (2026-05-01)

## Corpus Check
- 51 files · ~35,455 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 247 nodes · 309 edges · 35 communities detected
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]

## God Nodes (most connected - your core abstractions)
1. `render_recommendation_card()` - 13 edges
2. `main()` - 10 edges
3. `_display_name()` - 10 edges
4. `run_kmeans()` - 8 edges
5. `render_methodology_page()` - 8 edges
6. `HalalKMeans` - 7 edges
7. `build_viability()` - 7 edges
8. `main()` - 7 edges
9. `scripts/run_phase2.py — Risk viability merge, final_score + similarity rankings` - 7 edges
10. `build_entry_forecast()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `build_supply()` --calls--> `main()`  [INFERRED]
  src/halal_opportunity.py → scripts/run_phase1.py
- `build_gap()` --calls--> `main()`  [INFERRED]
  src/halal_opportunity.py → scripts/run_phase1.py
- `build_lisa()` --calls--> `main()`  [INFERRED]
  src/halal_spatial.py → scripts/run_phase3.py
- `build_demand()` --calls--> `test_mn22_latent_gt_revealed()`  [INFERRED]
  src/halal_demand.py → tests/test_halal_demand.py
- `build_demand()` --calls--> `main()`  [INFERRED]
  src/halal_demand.py → scripts/run_phase1.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (21): _generate_narrative(), Comparison view — side-by-side neighborhood analysis., render_comparison_view(), _deterministic_jitter(), Map view — plots NTA recommendations using borough centroids., render_map_view(), _borough(), _build_radar_chart() (+13 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (13): build_entry_forecast(), build_forecast(), _load_yearly_nta_signals(), HalalKMeans, run_kmeans(), main(), main(), test_build_entry_forecast_logic() (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (21): data/output/ — Derived CSVs powering Streamlit dashboards, frontend/app.py — Primary Streamlit recommender UX, frontend/methodology_content.py — Narrative + metrics for investor slides, scripts/run_phase1.py — Elbow analysis, clustering exports, centroid tables, scripts/run_phase2.py — Risk viability merge, final_score + similarity rankings, src/halal_kmeans.py — Custom NumPy k-means clustering + centroid labeling, src/halal_similarity.py — Cosine-similarity neighborhood look-alikes, Amanda Dong (yd2825) — Frontend/UX owner (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (14): filter_recommendations(), load_recommendations(), load_review_evidence_pool(), main(), Streamlit entrypoint — NYC Halal Restaurant Opportunity Finder., Yelp reviews with Gemini halal labels — used for qualitative evidence per NTA., Input form — borough, market type, and risk filters., render_input_form() (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.23
Nodes (9): build_gmm_risk(), build_viability(), _load_inspection_agg(), _zscore(), build_similarity(), main(), test_build_viability_columns(), test_risk_bucket_values() (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (12): data/processed/inspections.parquet — Per-inspection parquet with grades + nta_id, scripts/run_phase3.py — Ridge forecasts, GMM risk, final CSV outputs, src/halal_forecast.py — Temporal ridge models for halal chatter + entry dynamics, src/halal_risk.py — Inspection aggregation, viability heuristic, Gaussian-mixture risk, Catherine Yi (cgy2014) — Probabilistic overlays + forecasting, Gaussian Mixture Model (BIC-guided components) — soft-assignment NTA risk, Rationale: GMM chosen over hard clustering to capture heterogeneous NTA geometry, Rationale: ridge regression + k-fold CV chosen to stabilize moderately-sized longitudinal fits (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.31
Nodes (10): _load_elbow(), _load_final(), _load_phase1(), Methodology page — interactive explanation of the three-phase pipeline., Interactive weight slider — adjust 0.4/0.4/0.2 and see top NTAs change., _render_cluster_scatter(), _render_elbow_chart(), _render_formula_sandbox() (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (10): clip_review(), evidence_csv_path(), load_labeled_reviews(), nta_review_counts(), Load Yelp + Gemini labeled reviews for Streamlit neighborhood evidence., Stable key per venue: restaurant_id when present, else normalized business_name,, Returns a normalized dataframe for lookups, or None if file missing., Prefer explicit_halal → implicit_halal; then rating desc; at most one row per ve (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.36
Nodes (8): build_gap(), build_supply(), _demand(), _supply(), test_build_gap_columns(), test_build_gap_range(), test_build_supply_no_hygiene(), test_build_supply_no_nan()

### Community 9 - "Community 9"
Cohesion: 0.38
Nodes (8): build_demand(), build_latent_demand(), _load_raw_data(), _labels(), _reviews(), test_latent_demand_columns(), test_latent_demand_range(), test_mn22_latent_gt_revealed()

### Community 10 - "Community 10"
Cohesion: 0.48
Nodes (6): _load_phase1_results(), _load_phase2_results(), _load_phase3_results(), main(), Presentation-style slide deck page for the halal pipeline project., _render_intro_flowchart()

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (7): NYC Halal Opportunity Finder (Design Doc), NYC DOHMH CAMIS hygiene extracts — restaurant supply + cuisine source, data/processed/inspections.parquet — inspection-grade history aggregated to NTAs, Yelp review text + Gemini labels — demand signal source, NTA (Neighborhood Tabulation Area) — unit of analysis, Problem: halal restaurant location selection for NYC operators — information gap, Environment setup — Python 3.10+, venv, pip install -r requirements.txt

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (2): ModelConfig, test_custom_override()

### Community 13 - "Community 13"
Cohesion: 0.5
Nodes (4): build_lisa(), _load_centroids(), Parse NTA centroids from NTA boundaries CSV.          Parses WKT MULTIPOLYGON st, Compute Local Moran's I (LISA) for gap_score.          Args:         gap_df: Dat

### Community 14 - "Community 14"
Cohesion: 0.4
Nodes (1): Halal pipeline package.

### Community 15 - "Community 15"
Cohesion: 0.6
Nodes (5): data/raw/gemini_labels_full.csv — Gemini halal relevance labels, data/raw/yelp_reviews_with_zones.csv — Review text with NTA + Gemini join keys, src/halal_demand.py — Gemini-labeled Yelp text → NTA demand_score features, Siqi Zhu (sz3950) — Gemini/Yelp ingestion + demand-signal QA, pandas>=2.0.0

### Community 16 - "Community 16"
Cohesion: 0.5
Nodes (3): minmax(), Shared utilities — math helpers and domain constants for the halal pipeline., Min-max normalize a Series to [0, 1]. Returns 0.0 if constant or all-null.

### Community 18 - "Community 18"
Cohesion: 0.83
Nodes (3): main(), _summary(), _validate()

### Community 19 - "Community 19"
Cohesion: 0.5
Nodes (3): Data freshness helpers for the frontend., Render per-source freshness with live availability status., render_data_freshness()

### Community 20 - "Community 20"
Cohesion: 0.5
Nodes (3): Scenario controls — supports any cuisine type via free-text or dropdown., Render concept, price, and risk controls.  Supports any cuisine type., render_scenario_panel()

### Community 21 - "Community 21"
Cohesion: 0.67
Nodes (4): data/raw/restaurant_hygiene.csv — CAMIS universe with cuisine descriptors, scripts/check_camis_time.py — CAMIS timeline QA vs Yelp and parquet, src/halal_opportunity.py — CAMIS cuisines → halal supply rates, gaps, diversification, Harsh Agarwal (ha2957) — CAMIS supply metrics + reproducibility

### Community 22 - "Community 22"
Cohesion: 0.67
Nodes (1): Methodology content for the main Streamlit app.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Single source of truth for sidebar control keys and defaults.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Rich analytics for Tab 3.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Rich analytics for Tab 3.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Compute Local Moran's I (LISA) for gap_score.          Args:         gap_df: Dat

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Yelp reviews with Gemini halal labels — used for qualitative evidence per NTA.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Min-max normalize a Series to [0, 1]. Returns 0.0 if constant or all-null.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): A K-Means implementation built from scratch using NumPy.     Supports K-Means++

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): K-Means++ initialization.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Assign each sample to the nearest centroid.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Transform X to a cluster-distance space.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Engine for computing composite demand scores.     Integrates Revealed Demand (Ye

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): build_halal_scores.py — Planned merge façade (currently dormant)

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): scipy>=1.10.0

## Knowledge Gaps
- **58 isolated node(s):** `Shared utilities — math helpers and domain constants for the halal pipeline.`, `Min-max normalize a Series to [0, 1]. Returns 0.0 if constant or all-null.`, `Parse NTA centroids from NTA boundaries CSV.          Parses WKT MULTIPOLYGON st`, `Compute Local Moran's I (LISA) for gap_score.          Args:         gap_df: Dat`, `Streamlit entrypoint — NYC Halal Restaurant Opportunity Finder.` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (6 nodes): `ModelConfig`, `config.py`, `test_custom_override()`, `test_defaults_sane()`, `test_immutable()`, `test_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (5 nodes): `__init__.py`, `__init__.py`, `__init__.py`, `Halal pipeline package.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (3 nodes): `methodology_content.py`, `Methodology content for the main Streamlit app.`, `render_methodology_page()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `Single source of truth for sidebar control keys and defaults.`, `_form_keys.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Rich analytics for Tab 3.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Rich analytics for Tab 3.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Compute Local Moran's I (LISA) for gap_score.          Args:         gap_df: Dat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Yelp reviews with Gemini halal labels — used for qualitative evidence per NTA.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Min-max normalize a Series to [0, 1]. Returns 0.0 if constant or all-null.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `A K-Means implementation built from scratch using NumPy.     Supports K-Means++`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `K-Means++ initialization.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Assign each sample to the nearest centroid.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Transform X to a cluster-distance space.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Engine for computing composite demand scores.     Integrates Revealed Demand (Ye`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `build_halal_scores.py — Planned merge façade (currently dormant)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `scipy>=1.10.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 1` to `Community 8`, `Community 9`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `build_gmm_risk()` connect `Community 4` to `Community 1`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `render_recommendation_card()` (e.g. with `render_results_panel()` and `nta_review_counts()`) actually correct?**
  _`render_recommendation_card()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `main()` (e.g. with `inject_custom_theme()` and `render_input_form()`) actually correct?**
  _`main()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `_display_name()` (e.g. with `render_results_panel()` and `_generate_narrative()`) actually correct?**
  _`_display_name()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `run_kmeans()` (e.g. with `test_run_kmeans_shape()` and `test_confidence_range()`) actually correct?**
  _`run_kmeans()` has 5 INFERRED edges - model-reasoned connections that need verification._