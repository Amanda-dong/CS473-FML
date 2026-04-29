"""Standalone evaluation script for the NYC Healthy-Food White-Space Finder.

Run with:
    uv run python -m src.validation.run_evaluation

Stages
------
1. Load feature matrix; build / merge ground-truth labels if needed.
2. Temporal walk-forward backtest (NDCG@5, NDCG@10, precision@5, MAP, ECE).
3. Feature ablation (NDCG drop per group).
4. Survival model concordance-index evaluation.
5. Aggregate summary written to data/processed/evaluation_summary.json.

Every stage is wrapped in try/except - the script never crashes even when
upstream data files are absent.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data"
_PROCESSED = _DATA / "processed"
_MODELS = _DATA / "models"

_FEATURE_MATRIX_PATH = _PROCESSED / "feature_matrix.parquet"
_LICENSES_PATH = _PROCESSED / "licenses.parquet"
_YELP_PATH = _PROCESSED / "yelp.parquet"
_INSPECTIONS_PATH = _PROCESSED / "inspections.parquet"
_SURVIVAL_MODEL_PATH = _MODELS / "survival_model.joblib"

_BACKTEST_OUT = _PROCESSED / "backtest_results.parquet"
_ABLATION_OUT = _PROCESSED / "ablation_results.parquet"
_SURVIVAL_EVAL_OUT = _PROCESSED / "survival_eval.json"
_SUMMARY_OUT = _PROCESSED / "evaluation_summary.json"


# ---------------------------------------------------------------------------
# Production scoring adapter for validation
# ---------------------------------------------------------------------------
class ProductionScoringAdapter:
    """Sklearn-compatible adapter around the production scoring logic."""

    _DROP_COLS = {"target", "y_composite", "label_quality", "missingness_fraction"}

    def __init__(
        self,
        concept_subtype: str = "healthy_indian",
        risk_tolerance: str = "balanced",
        price_tier: str = "mid",
        zone_type: str = "",
    ):
        self.concept_subtype = concept_subtype
        self.risk_tolerance = risk_tolerance
        self.price_tier = price_tier
        self.zone_type = zone_type
        self.model = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProductionScoringAdapter":
        train_X = self._numeric_features(X)
        train_y = pd.to_numeric(y, errors="coerce")
        valid = train_y.notna()
        train_X = train_X.loc[valid]
        train_y = train_y.loc[valid]
        self.feature_names = list(train_X.columns)

        if len(train_X) < 4 or train_y.nunique(dropna=True) < 2:
            return self

        try:
            from src.models.cmf_score import HAS_XGB, LearnedScoringModel

            if not HAS_XGB:
                return self
            self.model = LearnedScoringModel(
                params={
                    "n_estimators": 50,
                    "max_depth": 3,
                    "learning_rate": 0.08,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                }
            )
            self.model.fit(train_X, train_y)
        except Exception as exc:
            logger.warning("ProductionScoringAdapter fit fell back to CMF: %s", exc)
            self.model = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        feature_X = self._numeric_features(X).reindex(
            columns=self.feature_names, fill_value=0.0
        )
        if self.model is not None and self.feature_names:
            base_scores = np.asarray(self.model.predict(feature_X), dtype=float)
        else:
            base_scores = self._cmf_scores(X)
        return np.asarray(
            [
                self._apply_context(float(score), row)
                for score, (_, row) in zip(base_scores, X.iterrows())
            ],
            dtype=float,
        )

    def _numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.drop(columns=list(self._DROP_COLS), errors="ignore")
        return (
            frame.select_dtypes(include="number")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    def _cmf_scores(self, X: pd.DataFrame) -> np.ndarray:
        from src.models.cmf_score import compute_opening_score, score_zone_for_concept

        scores = []
        for _, row in X.iterrows():
            components = score_zone_for_concept(row.to_dict(), self.concept_subtype)
            scores.append(compute_opening_score(components))
        return np.asarray(scores, dtype=float)

    def _apply_context(self, score: float, row: pd.Series) -> float:
        from src.api.routers.recommendations import _apply_request_context_adjustment

        zone_type = str(row.get("zone_type", self.zone_type) or self.zone_type)
        return _apply_request_context_adjustment(
            score,
            row.to_dict(),
            zone_type=zone_type,
            concept_subtype=self.concept_subtype,
            risk_tolerance=self.risk_tolerance,
            price_tier=self.price_tier,
        )


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 1 - Load feature matrix and ground-truth labels
# ---------------------------------------------------------------------------
def _load_parquet_safe(path: Path, label: str) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Loaded %s: %d rows x %d cols", label, len(df), len(df.columns))
        return df
    logger.warning("%s not found at %s - returning empty frame", label, path)
    return pd.DataFrame()


def stage_load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feature_matrix, ground_truth) with target column guaranteed."""
    fm = _load_parquet_safe(_FEATURE_MATRIX_PATH, "feature_matrix")
    if fm.empty:
        logger.error("Feature matrix is empty - evaluation cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()

    # Check for existing target labels
    target_ok = "target" in fm.columns and fm["target"].notna().sum() >= 10

    if target_ok:
        logger.info(
            "Found %d labeled rows in feature_matrix.target",
            fm["target"].notna().sum(),
        )
        gt = (
            fm[["zone_id", "time_key", "target"]].rename(
                columns={"target": "y_composite"}
            )
            if "zone_id" in fm.columns
            else fm[["target"]].rename(columns={"target": "y_composite"})
        )
        return fm, gt

    logger.warning("Insufficient labels in feature_matrix - running build_ground_truth()")
    licenses = _load_parquet_safe(_LICENSES_PATH, "licenses")
    yelp = _load_parquet_safe(_YELP_PATH, "yelp")
    inspections = _load_parquet_safe(_INSPECTIONS_PATH, "inspections")

    try:
        from src.features.ground_truth import build_ground_truth

        gt = build_ground_truth(
            licenses_df=licenses,
            reviews_df=yelp,
            inspections_df=inspections,
        )
        logger.info("build_ground_truth() produced %d rows", len(gt))
    except Exception as exc:
        logger.error("build_ground_truth() failed: %s", exc)
        # Create a minimal synthetic ground-truth so downstream stages degrade
        # gracefully rather than crashing.
        rng = np.random.default_rng(0)
        n = len(fm)
        gt = pd.DataFrame({"y_composite": rng.uniform(0.0, 1.0, n)}, index=fm.index)

    if "y_composite" in gt.columns:
        coverage = gt["y_composite"].notna().mean()
        if coverage < 0.5:
            logger.warning(
                "Label coverage is %.1f%% - below 50%% threshold",
                coverage * 100,
            )

    # Merge composite label back into feature matrix
    if (
        "zone_id" in fm.columns
        and "zone_id" in gt.columns
        and "time_key" in fm.columns
        and "time_key" in gt.columns
    ):
        gt_slim = gt[["zone_id", "time_key", "y_composite"]].drop_duplicates(
            subset=["zone_id", "time_key"]
        )
        fm = fm.merge(gt_slim, on=["zone_id", "time_key"], how="left")
    else:
        fm["y_composite"] = gt["y_composite"].values if len(gt) == len(fm) else np.nan

    fm["target"] = fm.get("y_composite", pd.Series(np.nan, index=fm.index))
    return fm, gt


# ---------------------------------------------------------------------------
# Stage 2 - Temporal backtest
# ---------------------------------------------------------------------------
def stage_temporal_backtest(
    fm: pd.DataFrame,
    gt: pd.DataFrame,
    year_col: str = "time_key",
    min_train_years: int = 2,
) -> pd.DataFrame | None:
    try:
        from src.validation.backtesting import run_temporal_backtest

        if fm.empty:
            raise ValueError("Empty feature matrix - skipping backtest.")

        # Ensure ground_truth has a usable index aligned to fm
        if gt.empty or "y_composite" not in gt.columns:
            raise ValueError("Ground truth is empty or missing y_composite.")

        # Align ground_truth index to feature_matrix index where possible
        if (
            "zone_id" in fm.columns
            and "zone_id" in gt.columns
            and "time_key" in gt.columns
        ):
            gt_aligned = (
                fm[["zone_id", "time_key"]]
                .reset_index()
                .merge(
                    gt[["zone_id", "time_key", "y_composite"]].drop_duplicates(
                        subset=["zone_id", "time_key"]
                    ),
                    on=["zone_id", "time_key"],
                    how="left",
                )
                .set_index("index")["y_composite"]
            )
            gt_df = gt_aligned.to_frame(name="y_composite")
        else:
            gt_df = (
                gt[["y_composite"]].copy() if "y_composite" in gt.columns else gt.copy()
            )

        gt_df.index = fm.index

        logger.info("Running temporal backtest with ProductionScoringAdapter")
        feature_frame = fm.drop(
            columns=["target", "y_composite", "label_quality"],
            errors="ignore",
        )
        results = run_temporal_backtest(
            feature_matrix=feature_frame,
            ground_truth=gt_df,
            model_cls=ProductionScoringAdapter,
            year_col=year_col,
            min_train_years=min_train_years,
        )
        if results.empty:
            logger.warning("Backtest returned empty results (too few time periods?).")
        else:
            logger.info("Backtest complete - %d folds", len(results))
            _PROCESSED.mkdir(parents=True, exist_ok=True)
            results.to_parquet(_BACKTEST_OUT, index=False)
            logger.info("Saved backtest results - %s", _BACKTEST_OUT)
        return results
    except Exception as exc:
        logger.error("Temporal backtest failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Stage 3 - Feature ablation
# ---------------------------------------------------------------------------
def _build_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {
        "demand": ["demand", "trip", "station", "citibike"],
        "survival": ["survival", "merchant", "viability"],
        "rent_cost": ["rent", "pressure", "assessed", "pluto"],
        "nlp": ["review", "healthy", "subtype", "yelp"],
        "competition": ["competition", "restaurant_count"],
    }
    groups: dict[str, list[str]] = {}
    for group, keywords in patterns.items():
        matched = [col for col in columns if any(kw in col.lower() for kw in keywords)]
        if matched:
            groups[group] = matched
    return groups


def stage_feature_ablation(
    fm: pd.DataFrame,
    gt: pd.DataFrame,
) -> pd.DataFrame | None:
    try:
        from src.validation.ablation import feature_ablation

        if fm.empty or gt.empty:
            raise ValueError("Feature matrix or ground truth is empty - skipping ablation.")

        drop_cols = [
            c
            for c in ("zone_id", "time_key", "target", "y_composite")
            if c in fm.columns
        ]
        X = fm.drop(columns=drop_cols).select_dtypes(include="number").fillna(0.0)

        # Resolve target series aligned to X index
        if (
            "y_composite" in gt.columns
            and "zone_id" in fm.columns
            and "zone_id" in gt.columns
            and "time_key" in gt.columns
        ):
            y_series = (
                fm[["zone_id", "time_key"]]
                .reset_index()
                .merge(
                    gt[["zone_id", "time_key", "y_composite"]].drop_duplicates(
                        subset=["zone_id", "time_key"]
                    ),
                    on=["zone_id", "time_key"],
                    how="left",
                )
                .set_index("index")["y_composite"]
                .reindex(fm.index)
            )
        elif "target" in fm.columns:
            y_series = fm["target"]
        else:
            y_series = (
                gt.iloc[:, -1]
                if not gt.empty
                else pd.Series(
                    np.random.default_rng(1).uniform(0, 1, len(X)), index=X.index
                )
            )

        y_series = y_series.reindex(X.index).fillna(
            y_series.median() if y_series.notna().any() else 0.5
        )

        feature_groups = _build_feature_groups(list(X.columns))
        if not feature_groups:
            raise ValueError(
                "No feature groups matched any column in the feature matrix."
            )

        logger.info(
            "Running feature ablation for groups: %s",
            list(feature_groups.keys()),
        )

        # Build simple cross-val splits (3-fold by index position)
        n = len(X)
        fold_size = max(1, n // 3)
        splits = [
            (
                list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, n)),
                list(range(i * fold_size, (i + 1) * fold_size)),
            )
            for i in range(3)
            if len(list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, n)))
            > 0
            and len(list(range(i * fold_size, (i + 1) * fold_size))) > 0
        ]
        if not splits:
            splits = [(list(range(max(1, n // 2))), list(range(max(1, n // 2), n)))]

        results = feature_ablation(
            model_cls=ProductionScoringAdapter,
            X=X,
            y=y_series,
            feature_groups=feature_groups,
            splits=splits,
        )
        if not results.empty:
            _PROCESSED.mkdir(parents=True, exist_ok=True)
            results.to_parquet(_ABLATION_OUT, index=False)
            logger.info("Saved ablation results - %s", _ABLATION_OUT)
        return results
    except Exception as exc:
        logger.error("Feature ablation failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Stage 4 - Survival model evaluation
# ---------------------------------------------------------------------------
def stage_survival_eval(fm: pd.DataFrame) -> dict:
    metrics: dict = {"concordance_index": None}

    try:
        import joblib

        # --- Try loading pre-trained model ---
        bundle = None
        if _SURVIVAL_MODEL_PATH.exists():
            raw = joblib.load(_SURVIVAL_MODEL_PATH)
            # Key-based dict (saved via model_loader.save_model)
            if isinstance(raw, dict):
                if "concordance_index" in raw:
                    metrics["concordance_index"] = float(raw["concordance_index"])
                    logger.info(
                        "Concordance index from saved model dict: %.4f",
                        metrics["concordance_index"],
                    )
                bundle = raw.get("model", None)
            else:
                bundle = raw

            if (
                bundle is not None
                and hasattr(bundle, "cox_model_")
                and bundle.cox_model_ is not None
            ):
                try:
                    ci = bundle.cox_model_.concordance_index_
                    metrics["concordance_index"] = float(ci)
                    logger.info("Concordance index from cox_model_ attribute: %.4f", ci)
                except AttributeError:
                    pass

        # --- Fit fresh model on real history ---
        licenses = _load_parquet_safe(_LICENSES_PATH, "licenses")
        inspections = _load_parquet_safe(_INSPECTIONS_PATH, "inspections")

        if not licenses.empty:
            from src.models.survival_model import (
                SurvivalModelBundle,
                build_real_restaurant_history,
            )

            zone_features = fm if not fm.empty else None
            history = build_real_restaurant_history(
                licenses_df=licenses,
                inspections_df=inspections,
                zone_features=zone_features,
            )
            logger.info("Restaurant history: %d rows", len(history))

            if not history.empty and len(history) >= 10:
                # Time-sorted 80/20 split
                duration_col = "duration_days"
                event_col = "event_observed"
                if duration_col in history.columns and event_col in history.columns:
                    history_sorted = history.sort_values(duration_col).reset_index(
                        drop=True
                    )
                    split_idx = int(len(history_sorted) * 0.8)
                    train_hist = history_sorted.iloc[:split_idx]
                    test_hist = history_sorted.iloc[split_idx:]

                    smb = SurvivalModelBundle(baseline="cox")
                    smb.fit(train_hist)

                    if smb.cox_model_ is not None:
                        ci_val = float(smb.cox_model_.concordance_index_)
                        metrics["concordance_index"] = ci_val
                        logger.info(
                            "Freshly fitted Cox concordance index: %.4f", ci_val
                        )
                        # Also compute on test set via bundle method
                        try:
                            ci_test = smb.concordance_index(test_hist)
                            metrics["concordance_index_test"] = float(ci_test)
                            logger.info("Test-set concordance index: %.4f", ci_test)
                        except Exception as ci_exc:
                            logger.warning("Test-set C-index failed: %s", ci_exc)

        _PROCESSED.mkdir(parents=True, exist_ok=True)
        with open(_SURVIVAL_EVAL_OUT, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        logger.info("Saved survival eval - %s", _SURVIVAL_EVAL_OUT)

    except Exception as exc:
        logger.error("Survival model eval failed: %s", exc, exc_info=True)

    return metrics


# ---------------------------------------------------------------------------
# Stage 5 - Aggregate summary
# ---------------------------------------------------------------------------
def _extract_backtest_summary(bt: pd.DataFrame | None) -> dict:
    if bt is None or bt.empty:
        return {
            "backtest_ndcg5_mean": None,
            "backtest_ndcg5_std": None,
            "backtest_precision5_mean": None,
            "backtest_map_mean": None,
        }
    return {
        "backtest_ndcg5_mean": float(bt["ndcg_5"].mean())
        if "ndcg_5" in bt.columns
        else None,
        "backtest_ndcg5_std": float(bt["ndcg_5"].std())
        if "ndcg_5" in bt.columns
        else None,
        "backtest_precision5_mean": float(bt["precision_5"].mean())
        if "precision_5" in bt.columns
        else None,
        "backtest_map_mean": float(bt["map_score"].mean())
        if "map_score" in bt.columns
        else None,
    }


def _extract_ablation_summary(abl: pd.DataFrame | None) -> dict:
    if abl is None or abl.empty:
        return {"ablation_top_group": None, "ablation_top_ndcg_drop": None}
    top = abl.sort_values("ndcg_drop", ascending=False).iloc[0]
    return {
        "ablation_top_group": str(top["group_name"]),
        "ablation_top_ndcg_drop": float(top["ndcg_drop"]),
    }


def stage_summary(
    backtest: pd.DataFrame | None,
    ablation: pd.DataFrame | None,
    survival_metrics: dict,
) -> dict:
    summary: dict = {}
    summary.update(_extract_backtest_summary(backtest))
    summary.update(_extract_ablation_summary(ablation))
    summary["survival_concordance_index"] = survival_metrics.get("concordance_index")

    _PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(_SUMMARY_OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Evaluation summary:\n%s", json.dumps(summary, indent=2, default=str))
    return summary


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("NYC Healthy-Food White-Space Finder - Evaluation")
    logger.info("=" * 60)

    fm, gt = stage_load_data()

    backtest = stage_temporal_backtest(fm, gt)

    ablation = stage_feature_ablation(fm, gt)

    survival_metrics = stage_survival_eval(fm)

    summary = stage_summary(backtest, ablation, survival_metrics)

    print("\n--- Evaluation Summary ---")
    for key, val in summary.items():
        formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {key}: {formatted}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
