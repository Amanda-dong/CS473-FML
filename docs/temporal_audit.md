# Temporal Notes

Updated: 2026-04-29

Current baseline: treat 2016+ as usable for city data (`licenses`, `inspections`, `acs`, `permits`, `311`).

`pluto` is static (not true year-by-year), so it is used as a static covariate.

`yelp`, `citibike`, `airbnb`, `reddit` depend on local file coverage. Keep them as enrichment, not hard requirements.

Model eval rule (keep this fixed):
- use temporal split only (blocked / rolling)
- do not use random split for headline result

Backend note:
- training window is read from `feature_matrix.parquet` `time_key`
- `/predict/cmf` returns it in `query.train_window`
- `/predict/trajectory` returns it in `train_window`
