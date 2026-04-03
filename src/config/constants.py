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
    "reddit",
    "complaints_311",
    "boundaries",
)

# ── Cuisine / concept subtypes ────────────────────────────────────────────────
# The system is cuisine-agnostic: any free-text concept subtype is accepted and
# normalised by canonical_subtype().  The list below is the *known* set used for
# frontend dropdowns and taxonomy keyword matching.  An unknown value is treated
# as a valid custom subtype rather than rejected.

HEALTHY_SUBTYPES = (
    # healthy / wellness-focused
    "salad_bowls",
    "mediterranean_bowls",
    "healthy_indian",
    "vegan_grab_and_go",
    "protein_forward_lunch",
    # broader cuisine types
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

# ── Model & evaluation configuration ────────────────────────────────────────
MODEL_CONFIG = {
    "scoring": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "survival": {
        "n_estimators": 100,
        "penalizer": 0.1,
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
}

MODEL_DIR = "data/models"
PROCESSED_DIR = "data/processed"
