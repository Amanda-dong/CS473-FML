"""Integration tests for full pipeline."""

import pandas as pd
from unittest.mock import patch
import scripts.run_full_pipeline as run_full_pipeline


def test_build_feature_matrix_stage_returns_null_free_dataframe(
    tmp_path, monkeypatch
) -> None:
    """Verify build_feature_matrix_stage produces null-free output."""

    # 1. Setup paths
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    monkeypatch.setattr(run_full_pipeline, "PROCESSED_DIR", processed_dir)

    # 2. Mock inputs to build_feature_matrix_stage
    # We need to ensure build_zone_year_matrix and fill_feature_matrix_nulls
    # are exercised and produce a valid output that results in a
    # null-free matrix.

    # Create sample ETL outputs
    mock_etl_outputs = {
        "licenses": pd.DataFrame(
            {
                "license_id": ["L1"],
                "event_date": ["2024-01-01"],
                "nta_id": ["BK0101"],
                "restaurant_id": ["R1"],
                "license_status": ["Active"],
            }
        ),
        "acs": pd.DataFrame(
            {
                "nta_id": ["BK01"],
                "population": [1000.0],
                "median_income": [50000.0],
                "rent_burden": [0.3],
            }
        ),
    }

    # We need to mock the feature builders that are imported inside
    # build_feature_matrix_stage or inside build_zone_year_matrix to avoid
    # complex dependencies.
    # Actually, we can just patch them to return simple DFs.

    with patch(
        "src.features.feature_matrix.build_zone_year_matrix"
    ) as mock_build_zones:
        # Return a simple DF with some NaNs to ensure they get filled
        mock_build_zones.return_value = pd.DataFrame(
            {
                "zone_id": ["bk-tandon"],
                "time_key": [2024],
                "feature_a": [1.0],
                "feature_b": [pd.NA],  # Should be filled
            }
        )

        with patch("src.features.ground_truth.build_ground_truth") as mock_gt:
            mock_gt.return_value = pd.DataFrame(
                {
                    "zone_id": ["bk-tandon"],
                    "time_key": [2024],
                    "y_composite": [0.5],
                    "label_quality": [1.0],
                }
            )

            # 3. Call the stage
            df = run_full_pipeline.build_feature_matrix_stage(mock_etl_outputs)

            # 4. Verify
            assert not df.isnull().values.any(), f"DF has nulls: {df.isnull().sum()}"
            assert "feature_a" in df.columns
            assert "feature_b" in df.columns
            assert "target" in df.columns
