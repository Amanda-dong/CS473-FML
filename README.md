# NYC Restaurant Intelligence Platform

ML Course · Spring 2026  
Team: Catherine, Harsh, Tony, Siqi, Amanda

## Project Overview
The NYC Restaurant Intelligence Platform predicts neighborhood-level opportunity for restaurant concepts by combining permits, licenses, consumer sentiment, mobility signals, rent trends, and competitive density.

## Repository Structure
```text
nyc-restaurant-intel/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── geojson/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── nlp/
│   ├── api/
│   └── validation/
├── frontend/
│   ├── app.py
│   └── components/
├── notebooks/
└── tests/
```

## Team Roles
- **Harsh & Siqi**: Backend / ML lead
- **Tony & Amanda**: Frontend / Data pipelines
- **Catherine**: Project lead, validation, integration, reporting

## Data Sources (planned)
- NYC OpenData permits and licenses
- Yelp business/review APIs
- Airbnb listings data
- Citi Bike public trip data
- Google Trends
- Reddit discussions
- NYC NTA GeoJSON boundaries

## Setup
1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy environment template:
   - `cp .env.example .env`
4. Run placeholder checks:
   - `pytest`

## Notes
- `data/raw/` is intended for downloaded source files and should not contain committed private datasets.
- This repository currently contains scaffold/stub modules for collaborative development.
