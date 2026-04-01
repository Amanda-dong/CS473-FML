"""Shared fixtures for the placeholder test suite."""

import pytest


@pytest.fixture
def sample_zone_features() -> dict[str, float]:
    """A compact feature dict used across model and feature tests."""

    return {
        "healthy_supply_ratio": 0.2,
        "subtype_gap": 0.8,
        "quick_lunch_demand": 0.9,
        "survival_score": 0.7,
        "rent_pressure": 0.2,
        "competition_score": 0.3,
    }
