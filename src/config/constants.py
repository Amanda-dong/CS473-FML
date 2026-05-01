"""Shared constants for the project."""

ACTIVE_DATASETS = (
    "permits",
    "licenses",
    "inspections",
    "acs",
    "pluto",
    "citibike",
    "airbnb",
    "yelp",
    "complaints_311",
    "boundaries",
)

# Canonical list of features in the 49-column Feature Matrix
FM_COLS = [
    "avg_confidence",
    "avg_rating",
    "dominant_subtype",
    "explicit_halal_review_count",
    "explicit_halal_share",
    "halal_count_static",
    "halal_fast_casual_share",
    "halal_negative_rate",
    "halal_positive_rate",
    "halal_related_review_count",
    "halal_related_share",
    "healthy_food_share",
    "healthy_indian_share",
    "implicit_halal_review_count",
    "implicit_halal_share",
    "implicit_to_explicit_ratio",
    "inspection_grade_avg",
    "inspection_grade_avg_static",
    "label_quality",
    "license_velocity",
    "mean_assessed_value",
    "median_income",
    "median_income_static",
    "mediterranean_bowls_share",
    "net_closes",
    "net_opens",
    "non_halal_negative_rate",
    "non_halal_positive_rate",
    "not_related_review_count",
    "not_related_share",
    "overall_negative_rate",
    "overall_positive_rate",
    "permit_velocity",
    "population",
    "population_static",
    "rent_burden",
    "rent_pressure",
    "restaurant_count",
    "restaurant_count_static",
    "salad_bowls_share",
    "smoothie_juice_share",
    "station_count",
    "subtype_gap",
    "target",
    "time_key",
    "total_review_count",
    "trip_count",
    "unique_restaurant_count",
    "zone_id",
]

HEALTHY_SUBTYPES = (
    "halal",
    "salad_bowls",
    "mediterranean_bowls",
    "healthy_indian",
    "vegan_grab_and_go",
    "protein_forward_lunch",
    "mexican",
    "chinese",
    "japanese",
    "korean",
    "thai",
    "italian",
    "greek",
    "middle_eastern",
    "caribbean",
    "ethiopian",
    "west_african",
    "american_comfort",
    "burgers",
    "pizza",
    "seafood",
    "ramen",
    "dim_sum",
    "bakery_cafe",
    "smoothie_juice",
)

MICROZONE_TYPES = (
    "campus_walkshed",
    "lunch_corridor",
    "transit_catchment",
    "business_district",
)

MODEL_CONFIG = {
    "scoring": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "features": [
            c
            for c in FM_COLS
            if c not in ("target", "time_key", "zone_id", "dominant_subtype")
        ],
    },
    "survival": {
        "n_estimators": 100,
        "penalizer": 0.1,
        "features": [
            c
            for c in FM_COLS
            if c not in ("target", "time_key", "zone_id", "dominant_subtype")
        ],
    },
    "evaluation": {
        "n_bootstrap": 1000,
        "confidence_level": 0.95,
        "n_cv_folds": 5,
    },
    "ground_truth_weights": (0.35, 0.25, 0.20, 0.20),
    "outlier_clip_sigma": 3.0,
    "temporal_val_year": 2022,
    "temporal_test_year": 2023,
    "temporal_data_start_year": 2020,
    "temporal_data_end_year": 2024,
}

MODEL_DIR = "data/models"
PROCESSED_DIR = "data/processed"
