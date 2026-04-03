"""ETL for Yelp business and review enrichment.

Set ``YELP_DATA_PATH`` env var to a CSV or JSON file from the Yelp Academic Dataset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="yelp",
    owner="data",
    spatial_unit="restaurant",
    time_grain="date",
    description="Review text and business metadata for enrichment only after coverage audit.",
    columns=("review_date", "business_id", "restaurant_id", "rating", "review_text"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# Synthetic data (Yelp dataset requires application/agreement)
# ---------------------------------------------------------------------------


def _load_local() -> pd.DataFrame:
    """Load Yelp data from a local file specified by env var."""
    path_str = os.environ.get("YELP_DATA_PATH", "")
    if not path_str:
        raise RuntimeError(
            "etl_yelp: YELP_DATA_PATH env var required. "
            "Download the Yelp Academic Dataset and set YELP_DATA_PATH to the review JSON/CSV."
        )
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"etl_yelp: YELP_DATA_PATH={path} does not exist")
    logger.info("etl_yelp: loading from local file %s", path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Load real Yelp data from local file. Raises if not configured."""
    return _load_local()
