"""Build the Phase 1 neighborhood finding dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
CENSUS_PATH = PROCESSED_DIR / "census_nta_features.csv"
HYGIENE_PATH = PROCESSED_DIR / "hygiene_nta_features.csv"
CITIBIKE_PATH = PROCESSED_DIR / "citibike_nta_features.csv"
YELP_PATH = PROCESSED_DIR / "yelp_nta_features.csv"
OUTPUT_PATH = PROCESSED_DIR / "phase1_neighborhood_finding.csv"

YELP_FILL_ZERO_COLUMNS = [
    "restaurant_count",
    "halal_count",
    "halal_share",
    "avg_rating",
    "total_review_count",
]


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["nta"] = frame["nta"].astype(str).str.strip().str.upper()
    return frame


def main() -> None:
    census = load_frame(CENSUS_PATH)[
        ["nta", "median_household_income", "population_16plus"]
    ]
    hygiene = load_frame(HYGIENE_PATH)[
        ["nta", "inspection_count", "avg_score", "critical_violation_rate"]
    ]
    citibike = load_frame(CITIBIKE_PATH)[
        ["nta", "trip_count", "unique_start_station_count"]
    ]
    yelp = load_frame(YELP_PATH)[
        [
            "nta",
            "restaurant_count",
            "halal_count",
            "halal_share",
            "avg_rating",
            "total_review_count",
        ]
    ]

    phase1 = census.merge(hygiene, on="nta", how="left")
    phase1 = phase1.merge(citibike, on="nta", how="left")
    phase1 = phase1.merge(yelp, on="nta", how="left")

    for column in YELP_FILL_ZERO_COLUMNS:
        phase1[column] = phase1[column].fillna(0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    phase1.to_csv(OUTPUT_PATH, index=False)

    print(f"rows: {len(phase1)}")
    print("columns:")
    print(list(phase1.columns))
    print("missing values by column:")
    print(phase1.isna().sum().to_string())
    print("first 10 rows:")
    print(phase1.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
