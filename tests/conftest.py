"""Shared fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_zone_features() -> dict[str, float]:
    """Compact feature dict used across model and feature tests."""
    return {
        "healthy_supply_ratio": 0.2,
        "subtype_gap": 0.8,
        "quick_lunch_demand": 0.9,
        "survival_score": 0.7,
        "rent_pressure": 0.2,
        "competition_score": 0.3,
        "license_velocity": 0.5,
        "healthy_review_share": 0.4,
    }


@pytest.fixture
def sample_license_events() -> pd.DataFrame:
    """20 rows of license events with required columns."""
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=20, freq="ME"),
            "restaurant_id": [f"R{i:03d}" for i in range(20)],
            "business_unique_id": [f"BU{i:03d}" for i in range(20)],
            "license_status": (["Active", "Issued", "Expired", "Inactive"] * 5),
            "nta_id": (["BK01", "BK02", "MN01", "MN02"] * 5),
            "category": ["restaurant"] * 20,
        }
    )


@pytest.fixture
def sample_pluto_frame() -> pd.DataFrame:
    """20 rows of PLUTO-style assessed-value data."""
    return pd.DataFrame(
        {
            "year": ([2022] * 10 + [2023] * 10),
            "nta_id": (["BK01", "BK02", "MN01", "MN02", "QN01"] * 4),
            "assessed_value": [500_000, 750_000, 1_200_000, 900_000, 400_000] * 4,
            "commercial_sqft": [2000, 3500, 5000, 4000, 1500] * 4,
        }
    )


@pytest.fixture
def sample_review_labels() -> pd.DataFrame:
    """20 rows of Gemini-style labeled reviews with zone/time keys."""
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(20)],
            "sentiment": (["positive", "neutral", "negative", "positive"] * 5),
            "concept_subtype": (["healthy_indian", "salad_bowls", "mediterranean_bowls", "vegan_grab_and_go"] * 5),
            "confidence": [0.9, 0.8, 0.7, 0.85] * 5,
            "zone_id": (["tandon-campus", "columbia-morn"] * 10),
            "time_key": ([2023] * 10 + [2024] * 10),
        }
    )


@pytest.fixture
def sample_restaurant_history() -> pd.DataFrame:
    """50 rows of test restaurant survival data."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "duration_days": rng.integers(30, 2001, size=n),
            "event_observed": rng.integers(0, 2, size=n),
            "rent_pressure": rng.uniform(0.0, 1.0, size=n),
            "competition_score": rng.uniform(0.0, 1.0, size=n),
            "inspection_grade_numeric": rng.uniform(1.0, 3.0, size=n),
        }
    )
