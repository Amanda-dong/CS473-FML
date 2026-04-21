"""Tests for pipeline orchestration helpers."""

import pandas as pd

from src.pipeline.orchestrator import ProjectPipeline
from src.pipeline.preflight import run_processed_data_preflight
from src.pipeline.stages import PIPELINE_STAGES


def test_pipeline_tracks_known_stages() -> None:
    """The pipeline should accept documented stage names."""

    pipeline = ProjectPipeline()
    pipeline.run_stage(PIPELINE_STAGES[0])
    assert PIPELINE_STAGES[0] in pipeline.completed_stages


def test_processed_preflight_reports_missing_artifacts(tmp_path) -> None:
    report = run_processed_data_preflight(tmp_path)

    assert not report.passed
    failed = {check.name for check in report.failed_checks}
    assert {"scoring_training", "survival_training", "embedding_corpus"} <= failed


def test_processed_preflight_passes_with_minimal_artifacts(
    tmp_path,
    sample_license_events: pd.DataFrame,
) -> None:
    feature_matrix = pd.DataFrame(
        {
            "zone_id": ["bk-tandon", "mn-columbia"],
            "time_key": [2024, 2024],
            "feature_a": [1.0, 2.0],
            "target": [0.8, 0.7],
            "label_quality": [1.0, 1.0],
        }
    )
    feature_matrix.to_parquet(tmp_path / "feature_matrix.parquet", index=False)

    sample_license_events.to_parquet(tmp_path / "licenses.parquet", index=False)
    inspections = pd.DataFrame(
        {
            "inspection_date": sample_license_events["event_date"],
            "restaurant_id": sample_license_events["restaurant_id"],
            "grade": ["A"] * len(sample_license_events),
            "critical_flag": ["Not Applicable"] * len(sample_license_events),
            "nta_id": sample_license_events["nta_id"],
            "cuisine_type": ["Unknown"] * len(sample_license_events),
            "zipcode": ["10001"] * len(sample_license_events),
        }
    )
    inspections.to_parquet(tmp_path / "inspections.parquet", index=False)

    reviews = pd.DataFrame(
        {
            "review_text": ["healthy bowls near campus", "fresh lunch option"],
            "restaurant_id": ["R000", "R001"],
            "review_date": ["2024-01-01", "2024-01-02"],
        }
    )
    reviews.to_parquet(tmp_path / "yelp.parquet", index=False)

    report = run_processed_data_preflight(
        tmp_path,
        min_scoring_rows=1,
        min_scoring_zones=1,
        min_embedding_rows=1,
        min_survival_rows=1,
        min_survival_events=1,
    )

    assert report.passed


def test_processed_preflight_handles_bad_survival_artifacts_without_raising(
    tmp_path,
) -> None:
    pd.DataFrame(
        {
            "zone_id": ["bk-tandon"],
            "time_key": [2024],
            "feature_a": [1.0],
            "target": [0.8],
            "label_quality": [1.0],
        }
    ).to_parquet(tmp_path / "feature_matrix.parquet", index=False)
    pd.DataFrame({"restaurant_id": ["r1"]}).to_parquet(
        tmp_path / "licenses.parquet", index=False
    )
    pd.DataFrame(
        {"review_text": ["healthy lunch"], "restaurant_id": ["r1"]}
    ).to_parquet(
        tmp_path / "yelp.parquet",
        index=False,
    )

    report = run_processed_data_preflight(
        tmp_path,
        min_scoring_rows=1,
        min_scoring_zones=1,
        min_embedding_rows=1,
        min_survival_rows=1,
        min_survival_events=1,
    )

    assert not report.passed
    survival_check = next(
        check for check in report.checks if check.name == "survival_training"
    )
    assert not survival_check.passed
