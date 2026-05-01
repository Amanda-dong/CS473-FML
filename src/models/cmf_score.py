"""Concept-Market-Fit (CMF) scoring — nuanced multi-signal opening score."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # type: ignore[import]

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    import shap  # type: ignore[import]

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

try:
    import joblib  # type: ignore[import]

    HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    HAS_JOBLIB = False


@dataclass(frozen=True)
class ScoreComponents:
    """All signals that feed the CMF opening score.

    Every component is normalised to [0, 1] before weighting.
    """

    # Demand-side signals
    healthy_gap_score: float  # under-supply of the concept relative to demand
    subtype_gap_score: float  # cuisine-specific gap within the broader category
    demand_signal_score: float = 0.5  # foot-traffic / daytime footfall proxy
    review_demand_score: float = 0.0  # healthy-demand share from review NLP

    # Supply-side / viability signals
    merchant_viability_score: float = 0.5  # survival model output
    license_velocity_score: float = 0.0  # net new licenses (positive = growing area)

    # Cost / risk signals
    competition_penalty: float = 0.0  # direct-competitor density
    rent_pressure_penalty: float = 0.0  # rent as fraction of max assessed value

    # Context signals
    transit_access_score: float = (
        0.5  # proximity to transit (walk-shed / transit catchment)
    )
    income_alignment_score: float = 0.5  # median income vs price tier alignment

    @property
    def composite_score(self) -> float:
        """Compute the overall weighted opening score."""
        return compute_opening_score(self)


def compute_opening_score(components: ScoreComponents) -> float:
    """Compute a transparent weighted opening score.

    Weights reflect the relative importance of each signal in predicting
    commercial success for a new restaurant concept in a micro-zone.

    Weight rationale:
    - subtype_gap_score (0.24): concept-specific whitespace is core thesis
    - merchant_viability_score (0.16): survival outlook
    - demand_signal_score (0.14): foot-traffic baseline
    - healthy_gap_score (0.12): overall under-supply in category
    - license_velocity_score (0.10): growing area reduces risk
    - review_demand_score (0.08): NLP sentiment confirms latent demand
    - transit_access_score (0.07): accessibility amplifies foot-traffic
    - income_alignment_score (0.05): concept/price fit to local income
    - competition_penalty (0.08): penalise saturated markets
    - rent_pressure_penalty (0.04): penalise high-rent zones
    """
    score = (
        components.subtype_gap_score * 0.24
        + components.merchant_viability_score * 0.16
        + components.demand_signal_score * 0.14
        + components.healthy_gap_score * 0.12
        + components.license_velocity_score * 0.10
        + components.review_demand_score * 0.08
        + components.transit_access_score * 0.07
        + components.income_alignment_score * 0.05
        - components.competition_penalty * 0.08
        - components.rent_pressure_penalty * 0.04
    )
    return round(float(score), 4)


def score_zone_for_concept(
    zone_features: dict, concept_subtype: str
) -> ScoreComponents:  # noqa: ARG001
    """Build ScoreComponents from a rich zone feature dict.

    Accepts any cuisine concept — concept_subtype is used for future
    subtype-specific weight tuning (logged for traceability).

    Expected keys (all optional, default to 0.5 / 0.0):
        halal_related_share, overall_positive_rate, subtype_gap, target,
        license_velocity, restaurant_count_static, rent_pressure,
        trip_count, median_income_static
    """

    def _safe(val: object, default: float) -> float:
        """Coerce val to float, returning default on None/NaN."""
        try:
            f = float(val)  # type: ignore[arg-type]
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    demand = _safe(zone_features.get("halal_related_share"), 0.5)
    gap = _safe(zone_features.get("subtype_gap"), 0.5)
    review_share = _safe(zone_features.get("overall_positive_rate"), 0.0)
    survival = _safe(zone_features.get("target"), 0.5)
    vel_raw = _safe(zone_features.get("license_velocity"), 0.0)
    # Normalise license velocity: sigmoid(vel) maps (−∞,+∞) → (0,1)
    vel_norm = 1.0 / (1.0 + math.exp(-vel_raw)) if vel_raw != 0.0 else 0.5
    # Normalise raw counts to [0, 1] using fixed reference ranges
    competition = min(
        _safe(zone_features.get("restaurant_count_static"), 0.0) / 50.0, 1.0
    )
    rent = _safe(zone_features.get("rent_pressure"), 0.0)  # already [0,1] in FM
    transit = min(_safe(zone_features.get("trip_count"), 0.0) / 200_000.0, 1.0)
    income = min(
        max(
            (_safe(zone_features.get("median_income_static"), 80_000.0) - 30_000.0)
            / 170_000.0,
            0.0,
        ),
        1.0,
    )

    return ScoreComponents(
        healthy_gap_score=gap,
        subtype_gap_score=gap,
        demand_signal_score=demand,
        review_demand_score=review_share,
        merchant_viability_score=survival,
        license_velocity_score=vel_norm,
        competition_penalty=competition,
        rent_pressure_penalty=rent,
        transit_access_score=transit,
        income_alignment_score=income,
    )


# ---------------------------------------------------------------------------
# Learned scoring model (Phase 4)
# ---------------------------------------------------------------------------


class LearnedScoringModel:
    """XGBoost-based zone attractiveness scorer."""

    def __init__(self, params: dict | None = None):
        self.model: "xgb.XGBRegressor | None" = None
        self.feature_names: list[str] = []
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> "LearnedScoringModel":
        """Train XGBoost regressor on feature matrix with composite outcome.

        Parameters
        ----------
        eval_set : list of (X, y) tuples, optional
            Validation data for early stopping. When provided, training stops
            if validation RMSE doesn't improve for 20 rounds.
        """
        if not HAS_XGB:
            raise ImportError("xgboost is required for LearnedScoringModel.fit()")
        self.feature_names = list(X.columns)
        self.model = xgb.XGBRegressor(**self.params)
        fit_kwargs: dict = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        n_bootstrap: int = 50,
        ci: float = 0.95,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap prediction intervals using subsets of trees.

        Returns (mean_pred, ci_lower, ci_upper).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")
        # Use different tree subsets (iteration ranges) for variance estimation
        booster = self.model.get_booster()
        n_trees = int(booster.num_boosted_rounds())
        rng = np.random.default_rng(seed)

        preds_all = []
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        for _ in range(n_bootstrap):
            # Sample ~80% of trees
            end = rng.integers(max(int(n_trees * 0.6), 1), n_trees + 1)
            preds_all.append(booster.predict(dmat, iteration_range=(0, int(end))))

        preds_arr = np.array(preds_all)
        mean_pred = preds_arr.mean(axis=0)
        alpha = (1 - ci) / 2
        ci_lower = np.percentile(preds_arr, 100 * alpha, axis=0)
        ci_upper = np.percentile(preds_arr, 100 * (1 - alpha), axis=0)
        return mean_pred, ci_lower, ci_upper

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted zone attractiveness scores."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values per feature per row."""
        if self.model is None:
            raise RuntimeError("Call fit() before explain().")
        if not HAS_SHAP:
            raise ImportError("shap is required for LearnedScoringModel.explain()")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)

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
    def load(cls, path: str) -> "LearnedScoringModel":
        """Load model from joblib."""
        if not HAS_JOBLIB:
            raise ImportError("joblib is required for load()")
        data = joblib.load(path)
        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance
