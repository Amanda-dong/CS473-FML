"""Tests for dataset scaffold modules."""

import pandas as pd
import pytest

import src.data.etl_runner as etl_runner
from src.data.audit import build_default_audit_rows
from src.data.base import DatasetSpec
from src.data.quality import prepare_training_frame, validate_dataset_contract
from src.data.registry import DATASET_REGISTRY


def test_registry_contains_core_datasets() -> None:
    """The scaffold should expose the main active sources."""

    for dataset_name in ("permits", "licenses", "inspections", "acs", "pluto"):
        assert dataset_name in DATASET_REGISTRY


def test_audit_rows_cover_the_registry() -> None:
    """The audit helper should expose one row per registered dataset."""

    assert len(build_default_audit_rows()) == len(DATASET_REGISTRY)


def test_validate_dataset_contract_rejects_missing_columns() -> None:
    spec = DatasetSpec(
        name="demo",
        owner="qa",
        spatial_unit="zone",
        time_grain="year",
        description="Demo dataset.",
        columns=("zone_id", "time_key"),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_contract(pd.DataFrame({"zone_id": ["z1"]}), spec)


def test_prepare_training_frame_deduplicates_and_filters_low_quality() -> None:
    frame = pd.DataFrame(
        {
            "zone_id": ["z1", "z1", "z2"],
            "time_key": [2024, 2024, 2024],
            "feature_a": [1.0, 2.0, float("inf")],
            "target": [0.6, 0.8, 0.5],
            "label_quality": [0.8, 0.9, 0.2],
        }
    )

    cleaned, report = prepare_training_frame(frame, target_col="target", min_label_quality=0.5)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "feature_a"] == pytest.approx(2.0)
    assert report.dropped_rows == 2


def test_etl_runner_skips_optional_sources_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeYelpModule:
        def run_etl(self, limit: int = 0) -> pd.DataFrame:  # noqa: ARG002
            raise RuntimeError("YELP_DATA_PATH env var required")

    monkeypatch.setattr(etl_runner, "_ETL_MODULES", {"yelp": FakeYelpModule()})

    results, status = etl_runner.run_all_etl()

    assert status["yelp"] == "skipped"
    assert results["yelp"].empty


def test_licenses_transform_keeps_business_unique_id_separate() -> None:
    from src.data.etl_licenses import transform

    raw = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01"],
            "business_unique_id": ["dca-42"],
            "license_status": ["Active"],
            "nta": ["BK09"],
            "business_category": ["Restaurant"],
        }
    )

    transformed = transform(raw)

    assert transformed.loc[0, "business_unique_id"] == "dca-42"
    assert pd.isna(transformed.loc[0, "restaurant_id"])
