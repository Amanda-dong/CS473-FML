# NYC Halal Restaurant Opportunity Finder

Data-driven Neighborhood Tabulation Area (NTA) short-lists that combine halal-related review signals (Yelp + Gemini labels), CAMIS-style restaurant supply proxies, NYC DOHMH inspection aggregates, clustering, probabilistic risk overlays, and lightweight forecasting for a merchant-facing Streamlit experience.

---

## Team

| Name             | NYU NetID | GitHub |
|------------------|-----------|--------|
| Amanda Dong      | `yd2825`  | [Amanda-dong](https://github.com/Amanda-dong) |
| Tony Zhao        | `sz3822`  | [Tonyzsp](https://github.com/Tonyzsp) |
| Harsh Agarwal    | `ha2957`  | [harshagarwalnyu](https://github.com/harshagarwalnyu) |
| Siqi Zhu         | `sz3950`  | [HelenZhutt](https://github.com/HelenZhutt) |
| Catherine Yi     | `cgy2014` | [catherinegyi](https://github.com/catherinegyi) |

---

## Overview

Analysis lives in `src/` and is executed phase-by-phase from `scripts/`. Outputs land in `data/output/` and fuel `frontend/` (Streamlit).

| Layer | Scripts / modules | What it produces |
|-------|-------------------|------------------|
| Phase 1 | `scripts/run_phase1.py`, `halal_kmeans.py` | Demand + supply gaps → clustered market types (`phase1_*.csv`) |
| Phase 2 | `scripts/run_phase2.py`, `halal_similarity.py` | Opportunity score, viability, cosine neighbors (`phase2_opportunity_scores.csv`) |
| Phase 3 | `scripts/run_phase3.py`, `halal_forecast.py`, `halal_risk.py` | GMM risk, ridge forecasts, adjusted ranking (`final_recommendations.csv`) |

Higher-level narratives also appear in [`PROPOSAL.md`](PROPOSAL.md) and [`DESIGN.md`](DESIGN.md).

---

## Data layout

Place inputs under:

- `data/raw/yelp_reviews_with_zones.csv` — review text with NTA + join keys aligned to Gemini labeling.
- `data/raw/gemini_labels_full.csv` — Gemini halal relevance labels keyed to reviews/business IDs.
- `data/raw/restaurant_hygiene.csv` — CAMIS universe with cuisine descriptors (tracked via Releases if omitted from git due to `.gitignore`).
- `data/processed/inspections.parquet` — per-inspection parquet with grades, violation flags, and `nta_id`.

Optional exploratory assets:

- `data/raw/yelp_business.csv` — enables richer maps inside `frontend/pages/2_Review_Zones_Demo.py`.

Outputs are regenerated under `data/output/` whenever you rerun the phases.

---

## Environment setup

Python 3.10+ recommended.

```bash
cd CS473-FML
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

`requirement.text` and `REQUIREMENTS.TXT` mirror `requirements.txt` for reviewers who want alternate filenames.

---

## Running the analytic pipeline

From the repository root (Windows PowerShell example):

```powershell
python scripts\run_phase1.py
python scripts\run_phase2.py
python scripts\run_phase3.py
```

`scripts/check_camis_time.py` prints timeline QA for Yelp vs CAMIS vs parquet-derived aggregates.

---

## Launching Streamlit apps

Primary recommender dashboard:

```powershell
streamlit run frontend\app.py
```

Presentation-style walkthrough (`frontend/pages/presentation.py`):

```powershell
streamlit run frontend\pages\presentation.py
```

Gemini labeling × micro-zone exploratory page (requires optional `yelp_business.csv`):

```powershell
streamlit run frontend\pages\2_Review_Zones_Demo.py
```

---

## Project layout cheat sheet

- `src/` — reusable building blocks (`halal_*` modules).
- `scripts/` — phase runners + QA utilities wired to filesystem paths relative to repo root.
- `frontend/` — Streamlit UX, reusable components (`components/`).
- `data/output/` — machine-generated CSVs powering the dashboards (tracked for demos when allowed).

Consult [`DESIGN.md`](DESIGN.md) for an annotated directory tree plus module ownership expectations.
