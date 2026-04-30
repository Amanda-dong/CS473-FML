"""Health endpoints for local development."""

import logging
from pathlib import Path
import pandas as pd
from fastapi import APIRouter

router = APIRouter(tags=["health"])

# Cache counts at startup
_FM_PATH = Path("data/processed/feature_matrix.parquet")
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


@router.get("/health")
async def health_check() -> dict[str, int | str]:
    """Return a minimal health payload for smoke tests."""

    return {
        "status": "ok",
        "fm_row_count": _COUNTS["fm_row_count"],
        "feature_count": _COUNTS["feature_count"],
    }
