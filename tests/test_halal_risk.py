import pytest
import pandas as pd
from src.halal_risk import build_viability


def test_build_viability_columns():
    df = build_viability()
    expected_columns = [
        "nta_id",
        "critical_rate",
        "grade_a_rate",
        "inspection_frequency",
        "viability_score",
        "risk_bucket",
    ]
    assert all(col in df.columns for col in expected_columns)


def test_viability_score_range():
    df = build_viability()
    assert df["viability_score"].min() >= 0
    assert df["viability_score"].max() <= 1


def test_risk_bucket_values():
    df = build_viability()
    unique_buckets = df["risk_bucket"].unique()
    for bucket in unique_buckets:
        assert bucket in ["Low", "Medium", "High"]
