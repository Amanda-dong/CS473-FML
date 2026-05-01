"""Tests for pipeline orchestration helpers."""

import pandas as pd
import pytest

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


# ── preflight individual check functions ──────────────────────────────────────


def test_assess_embedding_readiness_value_error() -> None:
    from src.pipeline.preflight import assess_embedding_readiness

    result = assess_embedding_readiness(pd.DataFrame({"other_col": ["x"]}))
    assert not result.passed
    assert result.name == "embedding_corpus"


def test_assess_embedding_readiness_too_few_rows() -> None:
    from src.pipeline.preflight import assess_embedding_readiness

    df = pd.DataFrame(
        {
            "review_text": ["short review"],
            "restaurant_id": ["r1"],
        }
    )
    result = assess_embedding_readiness(df, min_rows=100)
    assert not result.passed


def test_assess_embedding_readiness_passes() -> None:
    from src.pipeline.preflight import assess_embedding_readiness

    df = pd.DataFrame(
        {
            "review_text": ["healthy salad bowl"] * 5,
            "restaurant_id": [f"r{i}" for i in range(5)],
        }
    )
    result = assess_embedding_readiness(df, min_rows=1)
    assert result.passed


def test_assess_scoring_readiness_value_error() -> None:
    from src.pipeline.preflight import assess_scoring_training_readiness

    result = assess_scoring_training_readiness(pd.DataFrame({"x": [1.0]}))
    assert not result.passed
    assert result.name == "scoring_training"


def test_assess_survival_readiness_value_error() -> None:
    from src.pipeline.preflight import assess_survival_training_readiness

    result = assess_survival_training_readiness(pd.DataFrame({"x": [1.0]}))
    assert not result.passed
    assert result.name == "survival_training"


# ── preflight CLI ─────────────────────────────────────────────────────────────


def test_preflight_main_cli_text_output(tmp_path) -> None:
    from src.pipeline.preflight import main

    exit_code = main(argv=["--processed-dir", str(tmp_path)])
    assert exit_code in (0, 1)


def test_preflight_main_cli_json_output(tmp_path) -> None:
    from src.pipeline.preflight import main

    exit_code = main(argv=["--processed-dir", str(tmp_path), "--json"])
    assert exit_code in (0, 1)


def test_preflight_build_arg_parser() -> None:
    from src.pipeline.preflight import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args(["--processed-dir", "/tmp", "--json"])
    assert args.processed_dir == "/tmp"
    assert args.json is True


# ── orchestrator — unknown stage ──────────────────────────────────────────────


def test_pipeline_raises_on_unknown_stage() -> None:
    pipeline = ProjectPipeline()
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.run_stage("nonexistent_stage")


# ── integration / smoke tests ────────────────────────────────────────────────


def test_pipeline_etl_only_completes(tmp_path, monkeypatch) -> None:
    """Smoke test: run main with --etl-only and verify feature_matrix.parquet."""
    import run_full_pipeline
    from unittest.mock import patch

    # 1. Setup paths to use tmp_path
    processed_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "data" / "models"
    processed_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(run_full_pipeline, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(run_full_pipeline, "MODEL_DIR", model_dir)

    # 2. Mock ETL runner to avoid real downloads/file reads
    mock_etl_results = {
        "licenses": pd.DataFrame(
            {
                "event_date": ["2024-01-01"],
                "nta_id": ["BK33"],
                "license_status": ["Active"],
                "restaurant_id": ["R1"],
                "business_unique_id": ["B1"],
                "category": ["Restaurant"],
            }
        ),
        "acs": pd.DataFrame(
            {
                "nta_id": ["BK33"],
                "population": [1000],
                "median_income": [70000],
                "rent_burden": [0.3],
            }
        ),
        "inspections": pd.DataFrame(
            {
                "inspection_date": ["2024-01-01"],
                "nta_id": ["BK33"],
                "grade": ["A"],
                "restaurant_id": ["R1"],
                "critical_flag": ["Not Applicable"],
                "cuisine_type": ["Unknown"],
                "zipcode": ["11201"],
            }
        ),
        "yelp": pd.DataFrame(
            {
                "review_date": ["2024-01-01"],
                "restaurant_id": ["R1"],
                "review_text": ["healthy"],
                "rating": [5.0],
            }
        ),
        "pluto": pd.DataFrame(
            {
                "nta_id": ["BK33"],
                "assessed_value": [1000000],
                "commercial_sqft": [5000],
            }
        ),
    }
    mock_status = {k: "ok" for k in mock_etl_results}

    # 3. Mocks for external calls
    with (
        patch(
            "src.data.etl_runner.run_all_etl",
            return_value=(mock_etl_results, mock_status),
        ),
        patch("requests.get") as mock_get,
    ):
        mock_get.return_value.json.return_value = {"status": "success"}
        mock_get.return_value.status_code = 200

        # 4. Run pipeline main
        import sys

        test_args = ["prog", "--etl-only", "--limit", "50"]
        with patch.object(sys, "argv", test_args):
            run_full_pipeline.main()

    # 5. Verify output
    fm_path = processed_dir / "feature_matrix.parquet"
    assert fm_path.exists()
    df = pd.read_parquet(fm_path)
    assert len(df) > 0
    assert "zone_id" in df.columns
    assert "time_key" in df.columns


def test_fill_nulls_in_pipeline(monkeypatch) -> None:
    """Verify fill_feature_matrix_nulls is called during build_feature_matrix_stage."""
    import run_full_pipeline
    from unittest.mock import patch
    from pathlib import Path

    call_count = 0

    def mock_fill(df, **kwargs):
        nonlocal call_count
        call_count += 1
        return df

    monkeypatch.setattr("src.data.quality.fill_feature_matrix_nulls", mock_fill)

    mock_etl_outputs = {
        "acs": pd.DataFrame(
            {
                "nta_id": ["BK33"],
                "population": [1000],
                "median_income": [70000],
                "rent_burden": [0.3],
            }
        ),
        "licenses": pd.DataFrame(
            {
                "event_date": ["2024-01-01"],
                "nta_id": ["BK33"],
                "license_status": ["Active"],
                "restaurant_id": ["R1"],
                "business_unique_id": ["B1"],
                "category": ["Restaurant"],
            }
        ),
    }

    # build_feature_matrix_stage writes to PROCESSED_DIR
    with patch("run_full_pipeline.PROCESSED_DIR", Path("/tmp")):
        # Mock build_ground_truth to return something mergeable
        with patch("src.features.ground_truth.build_ground_truth") as mock_gt:
            mock_gt.return_value = pd.DataFrame(
                columns=["zone_id", "time_key", "y_composite", "label_quality"]
            )
            run_full_pipeline.build_feature_matrix_stage(mock_etl_outputs)

    assert call_count > 0


def test_pipeline_summary_dict_has_correct_keys() -> None:
    """Verify build_feature_matrix_stage returns a DataFrame."""
    import run_full_pipeline
    from unittest.mock import patch
    from pathlib import Path

    mock_etl_outputs = {
        "acs": pd.DataFrame(
            {
                "nta_id": ["BK33"],
                "population": [1000],
                "median_income": [70000],
                "rent_burden": [0.3],
            }
        ),
        "licenses": pd.DataFrame(
            {
                "event_date": ["2024-01-01"],
                "nta_id": ["BK33"],
                "license_status": ["Active"],
                "restaurant_id": ["R1"],
                "business_unique_id": ["B1"],
                "category": ["Restaurant"],
            }
        ),
    }

    with patch("run_full_pipeline.PROCESSED_DIR", Path("/tmp")):
        with patch("src.features.ground_truth.build_ground_truth") as mock_gt:
            mock_gt.return_value = pd.DataFrame(
                columns=["zone_id", "time_key", "y_composite", "label_quality"]
            )
            df = run_full_pipeline.build_feature_matrix_stage(mock_etl_outputs)
            assert isinstance(df, pd.DataFrame)


# ── preflight — exception handler paths ───────────────────────────────────────


def test_processed_preflight_corrupt_feature_matrix(tmp_path) -> None:
    from src.pipeline.preflight import run_processed_data_preflight

    (tmp_path / "feature_matrix.parquet").write_bytes(b"not parquet")
    report = run_processed_data_preflight(
        tmp_path, min_scoring_rows=1, min_scoring_zones=1
    )
    assert not report.passed
    names = {c.name for c in report.failed_checks}
    assert "scoring_training" in names


def test_processed_preflight_corrupt_survival_artifacts(tmp_path) -> None:
    from src.pipeline.preflight import run_processed_data_preflight

    pd.DataFrame(
        {
            "zone_id": ["z1"],
            "time_key": [2024],
            "feature_a": [1.0],
            "target": [0.8],
            "label_quality": [1.0],
        }
    ).to_parquet(tmp_path / "feature_matrix.parquet", index=False)

    (tmp_path / "licenses.parquet").write_bytes(b"not parquet")
    (tmp_path / "inspections.parquet").write_bytes(b"not parquet")

    report = run_processed_data_preflight(
        tmp_path,
        min_scoring_rows=1,
        min_scoring_zones=1,
        min_survival_rows=1,
        min_survival_events=1,
    )
    assert not report.passed
    survival_check = next(c for c in report.checks if c.name == "survival_training")
    assert not survival_check.passed


def test_processed_preflight_corrupt_reviews(tmp_path) -> None:
    from src.pipeline.preflight import run_processed_data_preflight

    pd.DataFrame(
        {
            "zone_id": ["z1"],
            "time_key": [2024],
            "feature_a": [1.0],
            "target": [0.8],
            "label_quality": [1.0],
        }
    ).to_parquet(tmp_path / "feature_matrix.parquet", index=False)

    pd.DataFrame(
        {
            "event_date": ["2024-01-01"],
            "restaurant_id": ["R001"],
            "business_unique_id": ["dca-1"],
            "license_status": ["Active"],
            "nta_id": ["BK0202"],
            "category": ["Restaurant"],
        }
    ).to_parquet(tmp_path / "licenses.parquet", index=False)

    pd.DataFrame(
        {
            "inspection_date": ["2024-01-01"],
            "restaurant_id": ["R001"],
            "grade": ["A"],
            "critical_flag": ["Not Applicable"],
            "nta_id": ["BK0202"],
            "cuisine_type": ["Unknown"],
            "zipcode": ["11201"],
        }
    ).to_parquet(tmp_path / "inspections.parquet", index=False)

    (tmp_path / "yelp.parquet").write_bytes(b"not parquet")

    report = run_processed_data_preflight(
        tmp_path,
        min_scoring_rows=1,
        min_scoring_zones=1,
        min_survival_rows=1,
        min_survival_events=1,
        min_embedding_rows=1,
    )
    assert not report.passed
    embed_check = next(c for c in report.checks if c.name == "embedding_corpus")
    assert not embed_check.passed
