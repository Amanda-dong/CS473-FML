# Halal NTA Recommender — Clean Pipeline

## Unit of analysis
NTA (4-char, e.g. BK09) — all data sources native NTA format, no crosswalk needed.

## Data sources
- Demand: data/raw/yelp_reviews_with_zones.csv + data/raw/gemini_labels_full.csv
- Supply + Risk: data/raw/restaurant_hygiene.csv + data/processed/inspections.parquet
- KMeans features: built from demand + supply signals

## Files
- src/halal_demand.py      -> demand_score per NTA (Yelp/Gemini)
- src/halal_opportunity.py -> gap_score per NTA (CAMIS supply)
- src/halal_risk.py        -> risk_score per NTA (CAMIS inspection)
- src/halal_kmeans.py      -> KMeans from scratch (numpy only, no sklearn)
- build_halal_scores.py    -> merges all layers, outputs Top 5 NTA

## Output
data/output/nta_scores.csv — one row per NTA, all scores + final rank
