"""Training script for survival model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import MODEL_DIR, PROCESSED_DIR
from src.data.quality import prepare_survival_history
from src.models.survival_model import (
    SurvivalModelBundle,
    build_real_restaurant_history,
)

try:
    import joblib  # type: ignore[import]

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

DATA_DIR = Path(PROCESSED_DIR)
_MODEL_DIR = Path(MODEL_DIR)


def _load_or_build_history() -> pd.DataFrame:
    """Load real restaurant history. Raises if data files are missing."""
    licenses_path = DATA_DIR / "licenses.parquet"
    inspections_path = DATA_DIR / "inspections.parquet"
    zone_path = DATA_DIR / "zone_features.parquet"

    if not licenses_path.exists() or not inspections_path.exists():
        raise FileNotFoundError(
            f"Real data required. Run the ETL pipeline first to generate:\n"
            f"  {licenses_path}\n  {inspections_path}\n"
            f"Use: uv run -m src.data.etl_runner"
        )

    licenses_df = pd.read_parquet(licenses_path)
    inspections_df = pd.read_parquet(inspections_path)

    # zone_features.parquet is keyed by NTA codes (e.g. "BK0202") — same format
    # used by the licenses nta_id column — so it joins directly.
    zone_features: pd.DataFrame | None = None
    if zone_path.exists():
        zone_features = pd.read_parquet(zone_path)

        # Enrich with NTA-level inspection quality (licenses lack per-restaurant
        # IDs so we can only join at the NTA level).
        if "nta_id" in inspections_df.columns and "grade" in inspections_df.columns:
            grade_num = {"A": 1.0, "B": 2.0, "C": 3.0}
            nta_grade = (
                inspections_df[inspections_df["grade"].isin(grade_num)]
                .assign(grade_num=lambda d: d["grade"].map(grade_num))
                .groupby("nta_id")["grade_num"]
                .mean()
                .reset_index()
                .rename(
                    columns={"nta_id": "zone_id", "grade_num": "nta_inspection_grade"}
                )
            )
            zone_features = zone_features.merge(nta_grade, on="zone_id", how="left")
            zone_features["nta_inspection_grade"] = zone_features[
                "nta_inspection_grade"
            ].fillna(zone_features["nta_inspection_grade"].median())

    print("Building restaurant history from real data...")
    history = build_real_restaurant_history(licenses_df, inspections_df, zone_features)
    history, _report = prepare_survival_history(history)
    return history


def _temporal_split(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar-based train/test split by cohort year to prevent temporal leakage.

    Splits by year_opened so test restaurants opened later than train restaurants.
    This prevents the calibration failure from duration-sorted splits (ECE 0.93+).
    """
    df = df.copy()
    if "year_opened" in df.columns and df["year_opened"].nunique() > 1:
        cutoff_year = df["year_opened"].quantile(1 - test_frac)
        train = df[df["year_opened"] <= cutoff_year].reset_index(drop=True)
        test = df[df["year_opened"] > cutoff_year].reset_index(drop=True)
        if len(train) > 0 and len(test) > 0:
            return train, test
    # Fallback: random split (no temporal leakage, just no cohort structure)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    split_idx = int(len(df) * (1 - test_frac))
    return (
        df.iloc[idx[:split_idx]].reset_index(drop=True),
        df.iloc[idx[split_idx:]].reset_index(drop=True),
    )


def _cross_validate_cindex(
    history: pd.DataFrame, baseline: str, n_folds: int = 5, seed: int = 42
) -> tuple[float, float]:
    """K-fold cross-validated C-index with 95% CI.

    Returns (mean_c_index, std_c_index).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(history))
    fold_size = len(history) // n_folds

    c_indices = []
    for i in range(n_folds):
        test_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.concatenate(
            [indices[: i * fold_size], indices[(i + 1) * fold_size :]]
        )
        train_df = history.iloc[train_idx]
        test_df = history.iloc[test_idx]

        model = SurvivalModelBundle(baseline=baseline)
        model.fit(train_df)
        c_indices.append(model.concordance_index(test_df))

    return float(np.mean(c_indices)), float(np.std(c_indices))


def train_and_evaluate() -> None:
    """Main training pipeline for survival models."""
    print("Loading / building survival data...")
    history = _load_or_build_history()
    print(f"  Total restaurants: {len(history)}")
    print(f"  Events observed: {int(history['event_observed'].sum())}")
    print(f"  Right-censored: {int((history['event_observed'] == 0).sum())}")

    train_df, test_df = _temporal_split(history)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    results: dict[str, dict[str, float]] = {}

    # --- Cox PH ---
    print("\nFitting Cox Proportional Hazards...")
    cox = SurvivalModelBundle(baseline="cox")
    cox.fit(train_df)
    c_cox = cox.concordance_index(test_df)
    brier_cox = cox.brier_score(test_df, times=[90, 180, 365, 730])
    print(f"  C-index (holdout): {c_cox:.4f}")
    print(f"  Brier scores:\n{brier_cox.to_string(index=False)}")

    # Cross-validated C-index
    print("  Running 5-fold CV for C-index...")
    cv_mean, cv_std = _cross_validate_cindex(history, "cox")
    print(f"  C-index (5-fold CV): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  95% CI: [{cv_mean - 1.96 * cv_std:.4f}, {cv_mean + 1.96 * cv_std:.4f}]")
    results["cox"] = {"c_index": c_cox, "cv_mean": cv_mean, "cv_std": cv_std}

    # Proportional hazards test
    ph_test = cox.test_proportional_hazards(train_df)
    if "error" not in ph_test:
        print(f"  PH assumption test passed: {ph_test.get('passed', 'unknown')}")
    else:
        print(f"  PH test: {ph_test['error']}")

    # --- RSF ---
    # Subsample to 8 000 rows for RSF — full 37K rows causes OOM on constrained
    # environments (RSF memory scales O(n × trees × features)).
    _RSF_SAMPLE = 8_000
    rng_rsf = np.random.default_rng(42)
    train_rsf = train_df.sample(
        min(_RSF_SAMPLE, len(train_df)), random_state=rng_rsf.integers(1 << 31)
    )
    print(f"\nFitting Random Survival Forest (n={len(train_rsf)} subsample)...")
    rsf = SurvivalModelBundle(baseline="rsf")
    rsf.fit(train_rsf)
    if rsf.uses_heuristic_:
        print("  (sksurv not available — fell back to heuristic)")
    c_rsf = rsf.concordance_index(test_df)
    brier_rsf = rsf.brier_score(test_df, times=[90, 180, 365, 730])
    print(f"  C-index (holdout): {c_rsf:.4f}")
    print(f"  Brier scores:\n{brier_rsf.to_string(index=False)}")

    if not rsf.uses_heuristic_:
        rsf_cv_sample = history.sample(min(_RSF_SAMPLE, len(history)), random_state=0)
        cv_mean_rsf, cv_std_rsf = _cross_validate_cindex(rsf_cv_sample, "rsf")
        print(f"  C-index (5-fold CV, subsample): {cv_mean_rsf:.4f} ± {cv_std_rsf:.4f}")
        results["rsf"] = {
            "c_index": c_rsf,
            "cv_mean": cv_mean_rsf,
            "cv_std": cv_std_rsf,
        }
    else:
        results["rsf"] = {"c_index": c_rsf}

    # --- Pick best model ---
    best_name = max(results, key=lambda k: results[k]["c_index"])
    best_model = cox if best_name == "cox" else rsf
    print(f"\nBest model: {best_name} (C-index={results[best_name]['c_index']:.4f})")

    # --- Calibration ---
    cal = best_model.calibration_data(test_df)
    if not cal.empty:
        print(f"\nCalibration data ({best_name}):")
        print(cal.to_string(index=False))
        if "bin_error" in cal.columns:
            ece = float((cal["bin_error"] * cal["count"]).sum() / cal["count"].sum())
            print(f"  Expected Calibration Error: {ece:.4f}")

    # --- Save ---
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "survival_model.joblib"
    if HAS_JOBLIB:
        joblib.dump(best_model, str(model_path))
        print(f"\nModel saved to {model_path}")
    else:
        print("\njoblib not available — model not saved to disk.")

    # --- Zone-level survival scores ---
    if "zone_id" in history.columns:
        zone_ids = history["zone_id"].unique()
        zone_rows = []
        for zid in zone_ids:
            zone_data = history[history["zone_id"] == zid]
            feature_cols = best_model.feature_columns_
            if feature_cols:
                zone_mean = zone_data[feature_cols].mean().to_frame().T
                risk = best_model.predict_risk(zone_mean).values[0]
                zone_rows.append(
                    {"zone_id": zid, "survival_score": round(1.0 - risk, 4)}
                )
        if zone_rows:
            zone_scores = pd.DataFrame(zone_rows)
            out_path = DATA_DIR / "zone_survival_scores.parquet"
            zone_scores.to_parquet(out_path, index=False)
            print(f"Zone survival scores saved to {out_path}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Model':<10} {'C-index':>10}")
    for name, metrics in results.items():
        print(f"{name:<10} {metrics['c_index']:>10.4f}")


if __name__ == "__main__":
    train_and_evaluate()
