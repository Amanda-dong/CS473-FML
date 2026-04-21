"""Feature ablation and baseline comparison studies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.backtesting import ndcg_at_k


def feature_ablation(
    model_cls,
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: dict[str, list[str]],
    splits: list[tuple],
) -> pd.DataFrame:
    """Remove each feature group, measure NDCG@5 drop.

    feature_groups example:
      {"demand": ["demand_signal", "quick_lunch_demand"],
       "nlp": ["healthy_review_share", "subtype_gap", "topic_0_share", ...],
       "survival": ["survival_score", "merchant_viability"],
       "cost": ["rent_pressure", "competition_score"]}

    Returns: group_name, ndcg_full, ndcg_ablated, ndcg_drop
    """
    def _avg_ndcg(X_sub: pd.DataFrame, y_s: pd.Series, splits_: list[tuple]) -> float:
        scores = []
        for train_idx, test_idx in splits_:
            model = model_cls()
            model.fit(X_sub.iloc[train_idx], y_s.iloc[train_idx])
            preds = model.predict(X_sub.iloc[test_idx])
            true = np.asarray(y_s.iloc[test_idx], dtype=float)
            scores.append(ndcg_at_k(np.asarray(preds, dtype=float), true, 5))
        return float(np.mean(scores)) if scores else 0.0

    full_score = _avg_ndcg(X, y, splits)

    records: list[dict] = []
    for group_name, cols in feature_groups.items():
        drop_cols = [c for c in cols if c in X.columns]
        if not drop_cols:
            continue
        X_ablated = X.drop(columns=drop_cols)
        ablated_score = _avg_ndcg(X_ablated, y, splits)
        records.append({
            "group_name": group_name,
            "ndcg_full": full_score,
            "ndcg_ablated": ablated_score,
            "ndcg_drop": full_score - ablated_score,
        })

    return pd.DataFrame(records)


def baseline_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    learned_model,
    splits: list[tuple],
) -> pd.DataFrame:
    """Compare learned model to baselines.

    Baselines:
    1. Random ranking
    2. Popularity (sort by license count / demand_signal)
    3. Heuristic (compute_opening_score from cmf_score.py)

    Returns: model_name, ndcg_5, ndcg_10, precision_5
    """
    from src.models.cmf_score import ScoreComponents, compute_opening_score

    results: list[dict] = []

    for model_name, score_fn in [
        ("learned", lambda X_tr, y_tr, X_te: _learned_predict(learned_model, X_tr, y_tr, X_te)),
        ("random", lambda X_tr, y_tr, X_te: np.random.default_rng(42).random(len(X_te))),
        ("popularity", lambda X_tr, y_tr, X_te: _popularity_scores(X_te)),
        ("heuristic", lambda X_tr, y_tr, X_te: _heuristic_scores(X_te)),
    ]:
        ndcg5_list, ndcg10_list, p5_list = [], [], []
        for train_idx, test_idx in splits:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            preds = np.asarray(score_fn(X_train, y_train, X_test), dtype=float)
            true = np.asarray(y_test, dtype=float)

            ndcg5_list.append(ndcg_at_k(preds, true, 5))
            ndcg10_list.append(ndcg_at_k(preds, true, 10))

            # precision@5
            true_top = set(str(i) for i in np.argsort(true)[::-1][:5])
            pred_ranking = [str(i) for i in np.argsort(preds)[::-1]]
            p5 = len(set(pred_ranking[:5]) & true_top) / 5
            p5_list.append(p5)

        results.append({
            "model_name": model_name,
            "ndcg_5": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
            "ndcg_10": float(np.mean(ndcg10_list)) if ndcg10_list else 0.0,
            "precision_5": float(np.mean(p5_list)) if p5_list else 0.0,
        })

    return pd.DataFrame(results)


def _learned_predict(model, X_train, y_train, X_test):
    """Fit and predict with the learned model."""
    import copy
    m = copy.deepcopy(model)
    m.fit(X_train, y_train)
    return m.predict(X_test)


def _popularity_scores(X: pd.DataFrame) -> np.ndarray:
    """Score by demand signal / license velocity if available."""
    if "demand_signal" in X.columns:
        return X["demand_signal"].values
    if "quick_lunch_demand" in X.columns:
        return X["quick_lunch_demand"].values
    return np.ones(len(X)) * 0.5


def permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric_fn=None,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance: shuffle each feature, measure metric drop.

    More robust than drop-column ablation because it accounts for
    multicollinearity and interaction effects.

    Parameters
    ----------
    model : fitted model with predict() method
    X_test, y_test : held-out test data
    metric_fn : callable(y_true, y_pred) -> float. Default: NDCG@5.
    n_repeats : number of shuffle repetitions per feature

    Returns DataFrame: feature, importance_mean, importance_std
    """
    if metric_fn is None:
        metric_fn = lambda y_true, y_pred: ndcg_at_k(  # noqa: E731
            np.asarray(y_pred, dtype=float),
            np.asarray(y_true, dtype=float),
            5,
        )

    rng = np.random.default_rng(seed)
    baseline_score = metric_fn(y_test, model.predict(X_test))

    records = []
    for col in X_test.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_score = metric_fn(y_test, model.predict(X_perm))
            drops.append(baseline_score - perm_score)
        records.append({
            "feature": col,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })

    result = pd.DataFrame(records).sort_values("importance_mean", ascending=False)
    return result.reset_index(drop=True)


def _heuristic_scores(X: pd.DataFrame) -> np.ndarray:
    """Score using compute_opening_score heuristic."""
    from src.models.cmf_score import ScoreComponents, compute_opening_score

    scores = []
    for _, row in X.iterrows():
        components = ScoreComponents(
            demand_signal_score=row.get("quick_lunch_demand", row.get("demand_signal", 0.5)),
            subtype_gap_score=row.get("subtype_gap", 0.5),
            merchant_viability_score=row.get("survival_score", 0.5),
            rent_pressure_penalty=row.get("rent_pressure", 0.3),
            competition_penalty=row.get("competition_score", 0.3),
            review_demand_score=row.get("healthy_review_share", 0.3),
            license_velocity_score=row.get("license_velocity", row.get("license_vel", 0.5)),
            transit_access_score=row.get("transit_access", 0.5),
            income_alignment_score=row.get("income_alignment", 0.5),
            healthy_gap_score=row.get("healthy_gap_score", 0.3),
        )
        scores.append(compute_opening_score(components))
    return np.array(scores)
