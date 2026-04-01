"""Survival model scaffold with a usable heuristic fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from lifelines import CoxPHFitter


@dataclass
class SurvivalModelBundle:
    """Scaffold for restaurant survival work."""

    baseline: str = "cox"
    duration_col: str = "duration_days"
    event_col: str = "event_observed"
    fitted_: bool = field(default=False, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    cox_model_: CoxPHFitter | None = field(default=None, init=False)
    uses_heuristic_: bool = field(default=False, init=False)

    def _select_numeric_features(self, restaurant_history: pd.DataFrame) -> list[str]:
        excluded = {self.duration_col, self.event_col}
        return [
            column
            for column in restaurant_history.select_dtypes(include=["number"]).columns
            if column not in excluded
        ]

    def fit(self, restaurant_history: pd.DataFrame) -> "SurvivalModelBundle":
        self.feature_columns_ = self._select_numeric_features(restaurant_history)
        required = {self.duration_col, self.event_col}
        if not required.issubset(restaurant_history.columns) or not self.feature_columns_:
            self.uses_heuristic_ = True
            self.fitted_ = True
            return self

        model_frame = restaurant_history[[self.duration_col, self.event_col, *self.feature_columns_]].copy()
        model_frame = model_frame.fillna(0.0)
        self.cox_model_ = CoxPHFitter()
        self.cox_model_.fit(model_frame, duration_col=self.duration_col, event_col=self.event_col)
        self.fitted_ = True
        return self

    def predict_risk(self, candidate_frame: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict_risk().")
        if self.uses_heuristic_ or self.cox_model_ is None:
            rent_pressure = (
                candidate_frame["rent_pressure"]
                if "rent_pressure" in candidate_frame
                else pd.Series([0.0] * len(candidate_frame), index=candidate_frame.index)
            )
            competition = (
                candidate_frame["competition_score"]
                if "competition_score" in candidate_frame
                else pd.Series([0.0] * len(candidate_frame), index=candidate_frame.index)
            )
            risk = (rent_pressure.astype(float) + competition.astype(float)) / 2.0
            return risk.clip(lower=0.0, upper=1.0).rename("closure_risk")

        score_frame = candidate_frame.reindex(columns=self.feature_columns_, fill_value=0.0).fillna(0.0)
        partial_hazard = self.cox_model_.predict_partial_hazard(score_frame)
        normalized = partial_hazard / partial_hazard.max() if float(partial_hazard.max()) else partial_hazard
        return normalized.rename("closure_risk")
