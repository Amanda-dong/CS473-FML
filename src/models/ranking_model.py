"""Small ranking helpers for concept-specific recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # type: ignore[import]

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    import joblib  # type: ignore[import]

    HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    HAS_JOBLIB = False

# rank:ndcg requires non-negative integer relevance grades; quartile bins
# give XGBoost enough resolution without amplifying noise from tied scores.
_NDCG_GRADE_BINS = 4


def rank_zones(
    scored_rows: Iterable[dict[str, float | str]],
    diversity_weight: float = 0.0,
) -> list[dict[str, float | str]]:
    """Sort scored rows by descending opportunity score with optional diversity.

    If diversity_weight > 0, we apply a small penalty to zones from a borough
    that is already well-represented in the top results.
    """
    sorted_rows = sorted(
        scored_rows,
        key=lambda row: float(row.get("opportunity_score", 0.0)),
        reverse=True,
    )

    if diversity_weight <= 0.0:
        return sorted_rows

    # Simple diversity re-ranking (MMR-lite)
    diverse_results: list[dict[str, float | str]] = []
    candidates = list(sorted_rows)
    borough_counts: dict[str, int] = {}

    while candidates and len(diverse_results) < len(sorted_rows):
        best_idx = -1
        best_score = -1e9

        for i, cand in enumerate(candidates):
            score = float(cand.get("opportunity_score", 0.0))
            borough = str(cand.get("borough", "Any"))

            # Penalty for borough redundancy
            count = borough_counts.get(borough, 0)
            penalty = diversity_weight * count * 0.05
            adjusted_score = score - penalty

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_idx = i

        selected = candidates.pop(best_idx)
        diverse_results.append(selected)
        borough = str(selected.get("borough", "Any"))
        borough_counts[borough] = borough_counts.get(borough, 0) + 1

    return diverse_results


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
        y_int = (
            pd.qcut(y, q=_NDCG_GRADE_BINS, labels=False, duplicates="drop")
            .fillna(0)
            .astype(int)
        )
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, y_int, group=group)
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
