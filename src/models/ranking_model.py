"""Small ranking helpers for concept-specific recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # type: ignore[import]

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import joblib  # type: ignore[import]

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def rank_zones(
    scored_rows: Iterable[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Sort scored rows by descending opportunity score."""

    return sorted(
        scored_rows,
        key=lambda row: float(row.get("opportunity_score", 0.0)),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Learned ranker (Phase 4)
# ---------------------------------------------------------------------------


class LearnedRanker:
    """LambdaMART ranking model via XGBoost."""

    def __init__(self, params: dict | None = None):
        self.model: "xgb.XGBRanker | None" = None
        self.feature_names: list[str] = []
        self.params = params or {
            "objective": "rank:ndcg",
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, group: list[int]) -> "LearnedRanker":
        """Train ranker. group = number of items per query group."""
        if not HAS_XGB:
            raise ImportError("xgboost is required for LearnedRanker.fit()")
        self.feature_names = list(X.columns)
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, y, group=group)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted relevance scores for ranking."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to joblib."""
        if not HAS_JOBLIB:
            raise ImportError("joblib is required for save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LearnedRanker":
        """Load model from joblib."""
        if not HAS_JOBLIB:
            raise ImportError("joblib is required for load()")
        data = joblib.load(path)
        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance
