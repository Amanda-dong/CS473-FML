"""Training script for the learned scoring model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import MODEL_DIR, PROCESSED_DIR
from src.data.quality import prepare_training_frame
from src.models.cmf_score import LearnedScoringModel, compute_opening_score, score_zone_for_concept
from src.models.ranking_model import LearnedRanker

DATA_DIR = Path(PROCESSED_DIR)
_MODEL_DIR = Path(MODEL_DIR)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and ground truth from data/processed/.

    Expects ``feature_matrix.parquet`` with a ``target`` column (composite
    outcome score) and optional ``year`` column for temporal splitting.
    Raises FileNotFoundError if the file is missing — run ETL first.
    """
    path = DATA_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Real data required at {path}. Run the ETL pipeline first:\n"
            f"  uv run -m src.data.etl_runner\n"
            f"Then build the feature matrix and ground truth."
        )
    df = pd.read_parquet(path)
    df, _report = prepare_training_frame(df, target_col="target")
    target_col = "target"
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_year: int = 2022,
    test_year: int = 2023,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """3-way temporal split: train (< val_year), val (val_year), test (>= test_year).

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    Validation set is used for early stopping in XGBoost.
    """
    time_col = "year" if "year" in X.columns else ("time_key" if "time_key" in X.columns else None)
    if time_col is not None:
        train_mask = X[time_col] < val_year
        val_mask = X[time_col] == val_year
        test_mask = X[time_col] >= test_year
        # Drop both time_col and zone_id (not features)
        drop_cols = [c for c in [time_col, "zone_id"] if c in X.columns]
        X_train = X.loc[train_mask].drop(columns=drop_cols)
        X_val = X.loc[val_mask].drop(columns=drop_cols)
        X_test = X.loc[test_mask].drop(columns=drop_cols)
        y_train, y_val, y_test = y.loc[train_mask], y.loc[val_mask], y.loc[test_mask]
    else:
        # Fallback: 60/20/20 split preserving order
        drop_cols = [c for c in ["zone_id"] if c in X.columns]
        X_clean = X.drop(columns=drop_cols) if drop_cols else X
        n = len(X_clean)
        s1, s2 = int(n * 0.6), int(n * 0.8)
        X_train, X_val, X_test = X_clean.iloc[:s1], X_clean.iloc[s1:s2], X_clean.iloc[s2:]
        y_train, y_val, y_test = y.iloc[:s1], y.iloc[s1:s2], y.iloc[s2:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute NDCG@k for a single query group."""
    order = np.argsort(-y_pred)[:k]
    dcg = np.sum(y_true[order] / np.log2(np.arange(2, len(order) + 2)))
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_order] / np.log2(np.arange(2, len(ideal_order) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _heuristic_scores(X: pd.DataFrame) -> np.ndarray:
    """Compute heuristic baseline scores for comparison."""
    scores = []
    for _, row in X.iterrows():
        components = score_zone_for_concept(row.to_dict(), "generic")
        scores.append(compute_opening_score(components))
    return np.array(scores)


def train_and_evaluate() -> None:
    """Main training pipeline."""
    print("Loading data...")
    X, y = load_data()

    print("Splitting data (temporal 3-way)...")
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- Scoring model (with early stopping on validation set) ---
    print("\nTraining LearnedScoringModel (XGBoost regressor)...")
    scorer = LearnedScoringModel()
    scorer.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    preds = scorer.predict(X_test)
    rmse = float(np.sqrt(np.mean((preds - y_test.values) ** 2)))
    print(f"  RMSE: {rmse:.4f}")

    ndcg5 = _ndcg_at_k(y_test.values, preds, 5)
    ndcg10 = _ndcg_at_k(y_test.values, preds, 10)
    print(f"  NDCG@5:  {ndcg5:.4f}")
    print(f"  NDCG@10: {ndcg10:.4f}")

    # --- Ranker ---
    print("\nTraining LearnedRanker (LambdaMART)...")
    # Treat entire test/train as single query group for simplicity
    ranker = LearnedRanker()
    ranker.fit(X_train, y_train, group=[len(X_train)])

    rank_preds = ranker.predict(X_test)
    rank_ndcg5 = _ndcg_at_k(y_test.values, rank_preds, 5)
    rank_ndcg10 = _ndcg_at_k(y_test.values, rank_preds, 10)
    print(f"  NDCG@5:  {rank_ndcg5:.4f}")
    print(f"  NDCG@10: {rank_ndcg10:.4f}")

    # --- Heuristic baseline ---
    print("\nHeuristic baseline (compute_opening_score)...")
    heuristic_preds = _heuristic_scores(X_test)
    heuristic_rmse = float(np.sqrt(np.mean((heuristic_preds - y_test.values) ** 2)))
    heuristic_ndcg5 = _ndcg_at_k(y_test.values, heuristic_preds, 5)
    heuristic_ndcg10 = _ndcg_at_k(y_test.values, heuristic_preds, 10)
    print(f"  RMSE:    {heuristic_rmse:.4f}")
    print(f"  NDCG@5:  {heuristic_ndcg5:.4f}")
    print(f"  NDCG@10: {heuristic_ndcg10:.4f}")

    # --- Save models ---
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scorer.save(str(_MODEL_DIR / "scoring_model.joblib"))
    ranker.save(str(_MODEL_DIR / "ranking_model.joblib"))
    print(f"\nModels saved to {_MODEL_DIR}/")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Metric':<15} {'Learned':>10} {'Ranker':>10} {'Heuristic':>10}")
    print(f"{'RMSE':<15} {rmse:>10.4f} {'N/A':>10} {heuristic_rmse:>10.4f}")
    print(f"{'NDCG@5':<15} {ndcg5:>10.4f} {rank_ndcg5:>10.4f} {heuristic_ndcg5:>10.4f}")
    print(f"{'NDCG@10':<15} {ndcg10:>10.4f} {rank_ndcg10:>10.4f} {heuristic_ndcg10:>10.4f}")


if __name__ == "__main__":
    train_and_evaluate()
