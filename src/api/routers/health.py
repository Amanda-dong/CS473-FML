"""Health endpoints for local development."""

import logging
from pathlib import Path
import pandas as pd
from fastapi import APIRouter
from src.schemas.results import HealthResponse

router = APIRouter(tags=["health"])

# Cache counts at startup
_FM_PATH = Path("data/processed/feature_matrix.parquet")
_SCORING_MODEL_PATH = Path("data/models/scoring_model.joblib")
_SURVIVAL_MODEL_PATH = Path("data/models/survival_model.joblib")

_COUNTS = {"fm_row_count": 0, "feature_count": 0}


def _refresh_counts():
    try:
        if _FM_PATH.exists():
            df = pd.read_parquet(_FM_PATH)
            _COUNTS["fm_row_count"] = len(df)
            _COUNTS["feature_count"] = len(df.columns)
    except Exception:
        logging.getLogger(__name__).warning("health: failed to read feature matrix")


_refresh_counts()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return an enriched health payload with data and model stats."""

    models_present = _SCORING_MODEL_PATH.exists() and _SURVIVAL_MODEL_PATH.exists()

    return HealthResponse(
        status="ok",
        fm_row_count=_COUNTS["fm_row_count"],
        feature_count=_COUNTS["feature_count"],
        feature_matrix_rows=_COUNTS["fm_row_count"],
        feature_matrix_cols=_COUNTS["feature_count"],
        model_files_present=models_present,
    )
