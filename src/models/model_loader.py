"""Lazy model loading with heuristic fallback and version tracking."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _candidate_paths(path_or_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Normalize one or more candidate paths while preserving order."""

    if isinstance(path_or_paths, (str, Path)):
        return [Path(path_or_paths)]
    return [Path(path) for path in path_or_paths]


def _first_existing_path(
    path_or_paths: str | Path | Iterable[str | Path],
) -> Path | None:
    """Return the first existing path from a candidate list."""

    for candidate in _candidate_paths(path_or_paths):
        if candidate.exists():
            return candidate
    return None


def save_model(model, path: str | Path, metadata: dict | None = None) -> None:
    """Save a model alongside a .meta.json with version tracking.

    Metadata includes: timestamp, feature_names, training_metrics, and any
    custom fields from the ``metadata`` dict.
    """
    try:
        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)

        meta = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_type": type(model).__name__,
        }
        if hasattr(model, "feature_names"):
            meta["feature_names"] = model.feature_names
        if hasattr(model, "feature_names_"):
            meta["feature_names"] = list(model.feature_names_)
        if metadata:
            meta.update(metadata)

        meta_path = p.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Model saved to %s with metadata", p)
    except Exception as e:
        logger.error("Failed to save model: %s", e)


def get_model_metadata(path: str | Path) -> dict | None:
    """Read .meta.json for a saved model. Returns None if not found."""
    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def get_model_version(path: str | Path) -> str:
    """Return a version string from model metadata, or 'unknown'."""
    meta = get_model_metadata(path)
    if meta:
        saved = meta.get("saved_at", "unknown")
        model_type = meta.get("model_type", "unknown")
        return f"{model_type}@{saved[:10]}"
    return "unknown"


def load_scoring_model(path: str | Path | Iterable[str | Path]):
    """Load LearnedScoringModel from joblib. Returns None if missing."""
    try:
        import joblib
        from src.models.cmf_score import LearnedScoringModel

        p = _first_existing_path(path)
        if p is not None:
            model = joblib.load(p)
            if isinstance(model, dict) and {
                "model",
                "feature_names",
                "params",
            }.issubset(model):
                hydrated = LearnedScoringModel(params=model["params"])
                hydrated.model = model["model"]
                hydrated.feature_names = list(model["feature_names"])
                model = hydrated
            version = get_model_version(p)
            logger.info("Scoring model loaded: %s", version)
            return model
    except Exception as e:
        logger.warning("Could not load scoring model: %s", e)
    return None


def load_survival_model(path: str | Path | Iterable[str | Path]):
    """Load SurvivalModelBundle from joblib. Returns None if missing."""
    try:
        import joblib

        p = _first_existing_path(path)
        if p is not None:
            model = joblib.load(p)
            version = get_model_version(p)
            logger.info("Survival model loaded: %s", version)
            return model
    except Exception as e:
        logger.warning("Could not load survival model: %s", e)
    return None


def load_feature_matrix(path: str | Path | Iterable[str | Path]):
    """Load pre-computed feature matrix from parquet. Returns None if missing."""
    try:
        import pandas as pd

        p = _first_existing_path(path)
        if p is not None:
            df = pd.read_parquet(p)
            logger.info(
                "Feature matrix loaded from %s: %d rows x %d cols",
                p,
                len(df),
                len(df.columns),
            )
            return df
    except Exception as e:
        logger.warning("Could not load feature matrix: %s", e)
    return None
