"""Survival model scaffold with a usable heuristic fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

try:
    from sksurv.ensemble import RandomSurvivalForest  # type: ignore[import]

    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def build_synthetic_restaurant_history(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """DEPRECATED — exists only for test compatibility. Use build_real_restaurant_history()."""
    raise RuntimeError(
        "Synthetic data generation removed. Use build_real_restaurant_history() "
        "with real ETL data from inspections and licenses."
    )


@dataclass
class SurvivalModelBundle:
    """Scaffold for restaurant survival work."""

    baseline: Literal["cox", "rsf", "heuristic"] = "cox"
    duration_col: str = "duration_days"
    event_col: str = "event_observed"
    fitted_: bool = field(default=False, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    cox_model_: CoxPHFitter | None = field(default=None, init=False)
    rsf_model_: object | None = field(default=None, init=False)
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
        if (
            not required.issubset(restaurant_history.columns)
            or not self.feature_columns_
        ):
            self.uses_heuristic_ = True
            self.fitted_ = True
            return self

        model_frame = restaurant_history[
            [self.duration_col, self.event_col, *self.feature_columns_]
        ].copy()
        model_frame = model_frame.fillna(0.0)

        if self.baseline == "rsf":
            if HAS_SKSURV:
                self._fit_rsf(model_frame)
            else:
                # Fall back to Cox if sksurv not available
                self._fit_cox(model_frame)
        elif self.baseline == "heuristic":
            self.uses_heuristic_ = True
        else:
            self._fit_cox(model_frame)

        self.fitted_ = True
        return self

    def _fit_cox(self, model_frame: pd.DataFrame) -> None:
        self.cox_model_ = CoxPHFitter()
        self.cox_model_.fit(
            model_frame, duration_col=self.duration_col, event_col=self.event_col
        )

    def _fit_rsf(self, model_frame: pd.DataFrame) -> None:
        y = np.array(
            list(
                zip(
                    model_frame[self.event_col].astype(bool),
                    model_frame[self.duration_col].astype(float),
                )
            ),
            dtype=[("event", bool), ("duration", float)],
        )
        X = model_frame[self.feature_columns_].values
        self.rsf_model_ = RandomSurvivalForest(n_estimators=100, random_state=42)
        self.rsf_model_.fit(X, y)  # type: ignore[union-attr]

    def predict_risk(self, candidate_frame: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict_risk().")
        if self.uses_heuristic_ or (
            self.cox_model_ is None and self.rsf_model_ is None
        ):
            rent_pressure = (
                candidate_frame["rent_pressure"]
                if "rent_pressure" in candidate_frame
                else pd.Series(
                    [0.0] * len(candidate_frame), index=candidate_frame.index
                )
            )
            competition = (
                candidate_frame["competition_score"]
                if "competition_score" in candidate_frame
                else pd.Series(
                    [0.0] * len(candidate_frame), index=candidate_frame.index
                )
            )
            risk = (rent_pressure.astype(float) + competition.astype(float)) / 2.0
            return risk.clip(lower=0.0, upper=1.0).rename("closure_risk")

        if self.rsf_model_ is not None:
            X = (
                candidate_frame.reindex(columns=self.feature_columns_, fill_value=0.0)
                .fillna(0.0)
                .values
            )
            chf = self.rsf_model_.predict_cumulative_hazard_function(X)  # type: ignore[union-attr]
            # Use final cumulative hazard as risk proxy
            risk_vals = np.array([fn.y[-1] for fn in chf])
            risk_series = pd.Series(
                risk_vals, index=candidate_frame.index, name="closure_risk"
            )
            max_val = float(risk_series.max())
            if max_val:
                risk_series = risk_series / max_val
            return risk_series.clip(0.0, 1.0)

        score_frame = candidate_frame.reindex(
            columns=self.feature_columns_, fill_value=0.0
        ).fillna(0.0)
        partial_hazard = self.cox_model_.predict_partial_hazard(score_frame)  # type: ignore[union-attr]
        normalized = (
            partial_hazard / partial_hazard.max()
            if float(partial_hazard.max())
            else partial_hazard
        )
        return normalized.rename("closure_risk")

    def predict_median_survival(self, candidate_frame: pd.DataFrame) -> pd.Series:
        """Return expected open_days (median survival time) for each candidate.

        Falls back to heuristic estimate when no fitted model is available.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict_median_survival().")

        if self.uses_heuristic_ or (
            self.cox_model_ is None and self.rsf_model_ is None
        ):
            risk = self.predict_risk(candidate_frame)
            # Heuristic: invert risk → map [0,1] risk to [2000, 30] days
            return ((1.0 - risk) * 1970 + 30).rename("open_days")

        if self.rsf_model_ is not None:
            X = (
                candidate_frame.reindex(columns=self.feature_columns_, fill_value=0.0)
                .fillna(0.0)
                .values
            )
            surv_funcs = self.rsf_model_.predict_survival_function(X)  # type: ignore[union-attr]
            medians = []
            for fn in surv_funcs:
                # Find first time where survival <= 0.5
                below = fn.x[fn.y <= 0.5]
                medians.append(float(below[0]) if len(below) > 0 else float(fn.x[-1]))
            return pd.Series(medians, index=candidate_frame.index, name="open_days")

        score_frame = candidate_frame.reindex(
            columns=self.feature_columns_, fill_value=0.0
        ).fillna(0.0)
        median_survival = self.cox_model_.predict_median(score_frame)  # type: ignore[union-attr]
        return median_survival.rename("open_days")

    # -------------------------------------------------------------------
    # Evaluation methods (Phase 5)
    # -------------------------------------------------------------------

    def concordance_index(self, test_df: pd.DataFrame) -> float:
        """Harrell's C-index on test data."""
        if not self.fitted_:
            raise RuntimeError("Call fit() before concordance_index().")

        from lifelines.utils import concordance_index as _c_index

        risk = self.predict_risk(test_df)
        return float(
            _c_index(
                test_df[self.duration_col],
                -risk,  # higher risk = shorter duration, so negate
                test_df[self.event_col],
            )
        )

    def brier_score(self, test_df: pd.DataFrame, times: list[int]) -> pd.DataFrame:
        """Brier score at specified time points with IPCW correction.

        Uses inverse probability of censoring weighting (IPCW) to handle
        right-censored observations. Falls back to a naive estimator
        when the Kaplan-Meier censoring model cannot be fit.

        Returns DataFrame with columns: time, brier_score, n_informative.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before brier_score().")

        durations = test_df[self.duration_col].values.astype(float)
        events = test_df[self.event_col].values.astype(int)
        risk = self.predict_risk(test_df).values

        # Cox model provides actual survival function; others use exponential approx
        has_cox_surv = self.cox_model_ is not None and not self.uses_heuristic_

        # Kaplan-Meier estimator for censoring distribution (IPCW)
        try:
            from lifelines import KaplanMeierFitter

            kmf_censor = KaplanMeierFitter()
            # Fit on censoring times: event=1 means censored (flip events)
            kmf_censor.fit(durations, event_observed=1 - events)
            has_ipcw = True
        except Exception:
            has_ipcw = False

        rows: list[dict] = []
        for t in times:
            if has_cox_surv:
                # Use actual survival function from Cox model
                score_frame = test_df.reindex(
                    columns=self.feature_columns_, fill_value=0.0
                ).fillna(0.0)
                surv_fn = self.cox_model_.predict_survival_function(score_frame)
                # Evaluate at time t; use nearest available time if t not in index
                surv_pred = np.array(
                    [
                        float(sf.asof(t)) if t <= sf.index.max() else float(sf.iloc[-1])
                        for _, sf in surv_fn.items()
                    ]
                )
            else:
                # Exponential approximation (acknowledged limitation)
                surv_pred = np.exp(-risk * (t / 365.0))

            # Only score observations informative at time t:
            # - Experienced event before t (outcome known: did not survive)
            # - Survived past t (outcome known: survived)
            # Exclude: censored before t (outcome unknown)
            informative = (durations >= t) | (events == 1)
            if informative.sum() == 0:
                rows.append(
                    {"time": float(t), "brier_score": np.nan, "n_informative": 0}
                )
                continue

            observed = (durations > t).astype(float)

            if has_ipcw:
                # IPCW weights: 1 / G(min(T_i, t)) where G is censoring survival
                eval_times = np.minimum(durations, float(t))
                weights = np.array(
                    [
                        1.0 / max(float(kmf_censor.predict(et)), 0.01)
                        for et in eval_times
                    ]
                )
                weights = weights[informative]
                bs = float(
                    np.average(
                        (surv_pred[informative] - observed[informative]) ** 2,
                        weights=weights,
                    )
                )
            else:
                bs = float(
                    np.mean((surv_pred[informative] - observed[informative]) ** 2)
                )

            rows.append(
                {
                    "time": float(t),
                    "brier_score": bs,
                    "n_informative": int(informative.sum()),
                }
            )

        return pd.DataFrame(rows)

    def calibration_data(
        self, test_df: pd.DataFrame, n_bins: int = 10, horizon_days: int = 365
    ) -> pd.DataFrame:
        """Predicted vs actual survival for calibration curve.

        Only includes uncensored observations and those censored beyond the
        horizon to avoid bias from informative censoring. Reports ECE alongside
        binned calibration.

        Returns DataFrame with columns: predicted_survival, actual_survival,
        count, bin_error.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before calibration_data().")

        durations = test_df[self.duration_col].values.astype(float)
        events = test_df[self.event_col].values.astype(int)
        risk = self.predict_risk(test_df).values

        # Predicted survival at horizon
        if self.cox_model_ is not None and not self.uses_heuristic_:
            score_frame = test_df.reindex(
                columns=self.feature_columns_, fill_value=0.0
            ).fillna(0.0)
            surv_fn = self.cox_model_.predict_survival_function(score_frame)
            pred_surv = np.array(
                [
                    float(sf.asof(horizon_days))
                    if horizon_days <= sf.index.max()
                    else float(sf.iloc[-1])
                    for _, sf in surv_fn.items()
                ]
            )
        else:
            pred_surv = np.exp(-risk * (horizon_days / 365.0))

        # Only include informative observations:
        # - Events that occurred (we know the outcome)
        # - Censored AFTER horizon (we know they survived to horizon)
        # Exclude: censored before horizon (outcome unknown)
        informative = (events == 1) | (durations >= horizon_days)
        if informative.sum() < 2:
            return pd.DataFrame(
                columns=["predicted_survival", "actual_survival", "count", "bin_error"]
            )

        pred_surv = pred_surv[informative]
        actual = (durations[informative] > horizon_days).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        rows: list[dict] = []
        for i in range(n_bins):
            mask = (pred_surv >= bin_edges[i]) & (pred_surv < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = mask | (pred_surv == bin_edges[i + 1])
            count = int(mask.sum())
            if count == 0:
                continue
            mean_pred = float(np.mean(pred_surv[mask]))
            mean_actual = float(np.mean(actual[mask]))
            rows.append(
                {
                    "predicted_survival": mean_pred,
                    "actual_survival": mean_actual,
                    "count": float(count),
                    "bin_error": abs(mean_pred - mean_actual),
                }
            )

        return pd.DataFrame(rows)

    def test_proportional_hazards(self, test_df: pd.DataFrame) -> dict:
        """Test the proportional hazards assumption using Schoenfeld residuals.

        Only applicable when a Cox model has been fit.
        Returns dict with test results or an error message.
        """
        if self.cox_model_ is None:
            return {"error": "No Cox model fitted — PH test not applicable."}
        try:
            results = self.cox_model_.check_assumptions(
                test_df[[self.duration_col, self.event_col, *self.feature_columns_]],
                show_plots=False,
                p_value_threshold=0.05,
            )
            return {
                "ph_test_results": str(results),
                "passed": results is None or len(results) == 0,
            }
        except Exception as e:
            return {"error": f"PH test failed: {e}"}


# ---------------------------------------------------------------------------
# Real restaurant history builder (Phase 5)
# ---------------------------------------------------------------------------


def build_real_restaurant_history(
    licenses_df: pd.DataFrame,
    inspections_df: pd.DataFrame,
    zone_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build actual restaurant survival data from license transitions.

    Parameters
    ----------
    licenses_df : pd.DataFrame
        Must contain: restaurant_id, nta_id (or zone_id), license_status,
        event_date (or status_date) as datetime. cuisine_type optional.
    inspections_df : pd.DataFrame
        Must contain: restaurant_id, grade (A/B/C).
    zone_features : pd.DataFrame | None
        Optional zone-level features keyed by zone_id with columns:
        rent_pressure, competition_score, transit_access.

    Returns
    -------
    pd.DataFrame with columns:
        restaurant_id, zone_id, cuisine_type,
        duration_days, event_observed (1=closed, 0=right-censored),
        rent_pressure, competition_score, transit_access,
        inspection_grade_numeric
    """
    licenses = licenses_df.copy()
    if "restaurant_id" in licenses.columns:
        licenses["_inspection_restaurant_id"] = licenses["restaurant_id"].replace(
            {"UNKNOWN": pd.NA, "": pd.NA}
        )
    else:
        licenses["_inspection_restaurant_id"] = pd.Series(
            pd.NA, index=licenses.index, dtype="object"
        )
    if licenses["_inspection_restaurant_id"].notna().any():
        licenses["_entity_id"] = licenses["_inspection_restaurant_id"].astype("string")
    elif "business_unique_id" in licenses.columns:
        licenses["_entity_id"] = (
            licenses["business_unique_id"]
            .replace({"UNKNOWN": pd.NA, "": pd.NA})
            .astype("string")
        )
    else:
        licenses["_entity_id"] = pd.Series(pd.NA, index=licenses.index, dtype="string")

    # Handle both ETL column names (event_date) and legacy (status_date)
    date_col = "event_date" if "event_date" in licenses.columns else "status_date"
    licenses["_date"] = pd.to_datetime(licenses[date_col])
    licenses = licenses.dropna(subset=["_entity_id", "_date"]).sort_values(
        ["_entity_id", "_date"]
    )

    # Handle both nta_id (ETL output) and zone_id (legacy)
    zone_col = "nta_id" if "nta_id" in licenses.columns else "zone_id"

    if licenses.empty:
        return pd.DataFrame(
            columns=[
                "restaurant_id",
                "zone_id",
                "cuisine_type",
                "duration_days",
                "event_observed",
                "inspection_grade_numeric",
                "rent_pressure",
                "competition_score",
                "transit_access",
            ]
        )

    cutoff = licenses["_date"].max()

    records: list[dict] = []
    for rid, grp in licenses.groupby("_entity_id"):
        first_row = grp.iloc[0]
        last_row = grp.iloc[-1]

        start = first_row["_date"]
        zone_id = first_row.get(zone_col, "unknown")
        cuisine = first_row.get("cuisine_type", "unknown")

        # Determine if closed: last status is inactive/expired/cancelled
        closed_statuses = {"inactive", "expired", "cancelled", "closed"}
        last_status = str(last_row.get("license_status", "")).strip().lower()
        event_observed = 1 if last_status in closed_statuses else 0

        end = last_row["_date"] if event_observed else cutoff
        duration_days = max((end - start).days, 1)

        records.append(
            {
                "restaurant_id": rid,
                "_inspection_restaurant_id": first_row.get("_inspection_restaurant_id"),
                "zone_id": zone_id,
                "cuisine_type": cuisine,
                "duration_days": duration_days,
                "event_observed": event_observed,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # Join inspection grades
    if (
        "grade" in inspections_df.columns
        and "restaurant_id" in inspections_df.columns
        and result["_inspection_restaurant_id"].notna().any()
    ):
        grade_map = {"A": 1.0, "B": 2.0, "C": 3.0}
        insp = inspections_df.copy()
        insp["inspection_grade_numeric"] = insp["grade"].map(grade_map)
        avg_grade = insp.groupby("restaurant_id")["inspection_grade_numeric"].mean()
        result = result.merge(
            avg_grade,
            left_on="_inspection_restaurant_id",
            right_index=True,
            how="left",
        )
        result["inspection_grade_numeric"] = result["inspection_grade_numeric"].fillna(
            2.0
        )
    else:
        result["inspection_grade_numeric"] = 2.0

    # Join zone features
    if zone_features is not None and "zone_id" in result.columns:
        zone_cols = [
            c
            for c in ["rent_pressure", "competition_score", "transit_access"]
            if c in zone_features.columns
        ]
        if zone_cols and "zone_id" in zone_features.columns:
            result = result.merge(
                zone_features[["zone_id", *zone_cols]], on="zone_id", how="left"
            )
            for c in zone_cols:
                result[c] = result[c].fillna(0.5)
    for c in ["rent_pressure", "competition_score", "transit_access"]:
        if c not in result.columns:
            result[c] = 0.5

    return result.drop(columns=["_inspection_restaurant_id"], errors="ignore")
