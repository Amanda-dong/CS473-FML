"""Tests for model placeholder helpers."""

import pandas as pd

from src.models.cmf_score import ScoreComponents, compute_opening_score
from src.models.ranking_model import rank_zones
from src.models.trajectory_model import TrajectoryClusteringModel


def test_opening_score_is_numeric() -> None:
    """The weighted score helper should return a float."""

    score = compute_opening_score(
        ScoreComponents(
            healthy_gap_score=0.8,
            subtype_gap_score=0.7,
            merchant_viability_score=0.6,
            competition_penalty=0.2,
        )
    )
    assert isinstance(score, float)


def test_ranking_orders_descending() -> None:
    """Higher scores should sort first."""

    rows = [{"zone_name": "A", "opportunity_score": 0.2}, {"zone_name": "B", "opportunity_score": 0.9}]
    ranked = rank_zones(rows)
    assert ranked[0]["zone_name"] == "B"


def test_trajectory_model_predicts_after_fit() -> None:
    """The placeholder clustering class should support fit/predict wiring."""

    model = TrajectoryClusteringModel().fit(pd.DataFrame({"value": [1, 2, 3]}))
    assert len(model.predict(pd.DataFrame({"value": [1, 2]}))) == 2
