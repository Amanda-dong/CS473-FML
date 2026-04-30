"""Tests for the API — sync smoke tests and async integration tests."""

from __future__ import annotations

import pytest
import pandas as pd

from src.api.main import app


# ── basic route checks ────────────────────────────────────────────────────────


def test_api_title() -> None:
    assert app.title == "NYC Restaurant Intelligence Platform API"


def test_api_version() -> None:
    assert app.version == "0.2.0"


def test_api_has_routes() -> None:
    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths
    assert "/datasets" in route_paths
    assert "/predict/cmf" in route_paths
    assert "/predict/trajectory" in route_paths


# ── async integration tests (httpx ASGI) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_datasets_endpoint_returns_list() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]


@pytest.mark.asyncio
async def test_predict_cmf_healthy_indian() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/cmf",
            json={"concept_subtype": "healthy_indian", "limit": 3},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 3


@pytest.mark.asyncio
async def test_predict_cmf_ramen() -> None:
    """CMF endpoint must handle non-healthy cuisine types."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/cmf",
            json={"concept_subtype": "ramen", "limit": 2},
        )
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert len(recs) == 2
    assert all(r["concept_subtype"] == "ramen" for r in recs)


@pytest.mark.asyncio
async def test_predict_cmf_custom_cuisine() -> None:
    """Free-text custom cuisine should not raise an error."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/cmf",
            json={"concept_subtype": "Peruvian Ceviche", "limit": 2},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_predict_cmf_returns_sorted_scores() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/cmf",
            json={"concept_subtype": "mediterranean_bowls", "limit": 5},
        )
    recs = resp.json()["recommendations"]
    scores = [r["opportunity_score"] for r in recs]
    assert scores == sorted(scores, reverse=True), (
        "Recommendations must be sorted by score"
    )


@pytest.mark.asyncio
async def test_predict_cmf_confidence_buckets() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/cmf", json={"concept_subtype": "salad_bowls", "limit": 5}
        )
    recs = resp.json()["recommendations"]
    for r in recs:
        assert r["confidence_bucket"] in ("high", "medium", "low")


@pytest.mark.asyncio
async def test_predict_trajectory_returns_cluster() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/trajectory", json={"concept_subtype": "korean"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "trajectory_cluster" in data
    assert "concept_subtype" in data


@pytest.mark.asyncio
async def test_predict_cmf_borough_filter() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/cmf",
            json={
                "concept_subtype": "vegan_grab_and_go",
                "borough": "Brooklyn",
                "limit": 3,
            },
        )
    assert resp.status_code == 200


def test_score_with_learned_model_uses_latest_time_key_and_survival_score() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    class DummyScoringModel:
        feature_names = ["healthy_review_share", "survival_score"]

        def predict(self, frame: pd.DataFrame) -> list[float]:
            return frame["healthy_review_share"].tolist()

    feature_matrix = pd.DataFrame(
        {
            "zone_id": ["bk-tandon", "bk-tandon"],
            "time_key": [2022, 2024],
            "healthy_review_share": [0.1, 0.9],
            "survival_score": [0.3, 0.75],  # latest row (2024) = 0.75
        }
    )

    rec = _score_with_learned_model(
        "bk-tandon",
        "NYU Tandon / MetroTech",
        "salad_bowls",
        feature_matrix,
        DummyScoringModel(),
        None,
    )

    assert rec is not None
    assert rec.opportunity_score == pytest.approx(0.9)
    assert rec.survival_risk == pytest.approx(0.25)  # 1.0 - 0.75
    assert rec.scoring_path == "learned"


def test_score_with_learned_model_applies_request_context() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    class DummyScoringModel:
        feature_names = ["rent_pressure", "competition_score", "income_alignment"]

        def predict(self, frame: pd.DataFrame) -> list[float]:
            return [0.5]

    feature_matrix = pd.DataFrame(
        {
            "zone_id": ["bk-tandon"],
            "time_key": [2024],
            "rent_pressure": [0.7],
            "competition_score": [0.6],
            "income_alignment": [0.8],
        }
    )

    conservative = _score_with_learned_model(
        "bk-tandon",
        "NYU Tandon / MetroTech",
        "healthy_indian",
        feature_matrix,
        DummyScoringModel(),
        None,
        zone_type="campus_walkshed",
        risk_tolerance="conservative",
        price_tier="premium",
    )
    aggressive = _score_with_learned_model(
        "bk-tandon",
        "NYU Tandon / MetroTech",
        "healthy_indian",
        feature_matrix,
        DummyScoringModel(),
        None,
        zone_type="campus_walkshed",
        risk_tolerance="aggressive",
        price_tier="budget",
    )

    assert conservative is not None
    assert aggressive is not None
    assert aggressive.opportunity_score > conservative.opportunity_score


# ── _confidence_bucket — all thresholds ──────────────────────────────────────


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.61, "high"),
        (1.0, "high"),
        (0.60, "medium"),
        (0.41, "medium"),
        (0.40, "low"),
        (0.2, "low"),
        (0.0, "low"),
    ],
)
def test_confidence_bucket(score: float, expected: str) -> None:
    from src.api.routers.recommendations import _confidence_bucket

    assert _confidence_bucket(score) == expected


# ── _get_zone_type_clusters ───────────────────────────────────────────────────


def test_get_zone_type_clusters_returns_dict() -> None:
    from src.api.routers.recommendations import _get_zone_type_clusters

    result = _get_zone_type_clusters("healthy_indian", "medium", "mid")
    assert isinstance(result, dict)
    # Should have entries for all zone types present in _NYC_ZONES
    for zt in (
        "campus_walkshed",
        "lunch_corridor",
        "transit_catchment",
        "business_district",
    ):
        assert zt in result
    # Each value should be a string cluster label
    for label in result.values():
        assert isinstance(label, str)


def test_get_zone_type_clusters_aggressive_risk() -> None:
    from src.api.routers.recommendations import _get_zone_type_clusters

    result = _get_zone_type_clusters("ramen", "aggressive", "premium")
    assert isinstance(result, dict)
    assert len(result) > 0


# ── _score_one ────────────────────────────────────────────────────────────────


def test_score_one_returns_zone_recommendation() -> None:
    from src.api.routers.recommendations import _score_one

    result = _score_one(
        "bk-tandon",
        "campus_walkshed",
        "NYU Area",
        "healthy_indian",
        "medium",
        "mid",
    )
    assert result.zone_id == "bk-tandon"
    assert 0.0 <= result.opportunity_score <= 2.0
    assert result.concept_subtype == "healthy_indian"
    assert result.confidence_bucket in ("high", "medium", "low")
    assert isinstance(result.positives, list)
    assert isinstance(result.risks, list)


def test_score_one_unknown_zone_uses_default_seed() -> None:
    """A zone_id not in _ZONE_SEEDS uses the default seed tuple."""
    from src.api.routers.recommendations import _score_one

    result = _score_one(
        "zz-unknown",
        "transit_catchment",
        "Unknown Zone",
        "ramen",
        "conservative",
        "budget",
    )
    assert result.zone_id == "zz-unknown"
    assert 0.0 <= result.opportunity_score <= 2.0


def test_score_one_price_and_risk_adjustments() -> None:
    """Conservative+premium suppresses survival → higher risk than aggressive+budget."""
    from src.api.routers.recommendations import _score_one

    rec_premium = _score_one(
        "bk-tandon", "campus_walkshed", "L", "salad_bowls", "conservative", "premium"
    )
    rec_budget = _score_one(
        "bk-tandon", "campus_walkshed", "L", "salad_bowls", "aggressive", "budget"
    )
    # _RISK_ADJUST[conservative]=-0.06, _PRICE_ADJUST[premium]=-0.04 → lower survival_score
    # _RISK_ADJUST[aggressive]=+0.06, _PRICE_ADJUST[budget]=+0.04 → higher survival_score
    assert rec_premium.survival_risk > rec_budget.survival_risk


def test_get_app_settings() -> None:
    from src.api.deps import get_app_settings
    from src.config import Settings

    settings = get_app_settings()
    assert isinstance(settings, Settings)


def test_score_with_learned_model_index_lookup() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    class DummyScoringModel:
        def predict(self, frame: pd.DataFrame) -> list[float]:
            return [0.75]

    feature_matrix = pd.DataFrame(
        {"feat1": [0.5]}, index=pd.Index(["bk-tandon"], name="zone_id")
    )
    rec = _score_with_learned_model(
        "bk-tandon", "Label", "subtype", feature_matrix, DummyScoringModel(), None
    )
    assert rec is not None
    assert rec.zone_id == "bk-tandon"


def test_score_with_learned_model_missing_zone() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    feature_matrix = pd.DataFrame({"zone_id": ["other"], "feat1": [0.5]})
    rec = _score_with_learned_model(
        "missing", "Label", "subtype", feature_matrix, None, None
    )
    assert rec is None


def test_score_with_learned_model_predict_fallback() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    class DummyScoringModel:
        def predict(self, frame: pd.DataFrame) -> list[float]:
            return [0.5]

    feature_matrix = pd.DataFrame(
        {"zone_id": ["bk-tandon"], "feat1": [0.5], "survival_score": [0.8]}
    )
    rec = _score_with_learned_model(
        "bk-tandon", "Label", "subtype", feature_matrix, DummyScoringModel(), None
    )
    assert rec.survival_risk == pytest.approx(0.2)  # 1.0 - 0.8


def test_score_with_learned_model_shap_tree_explainer() -> None:
    import numpy as np
    import xgboost as xgb
    from src.api.routers.recommendations import _score_with_learned_model

    X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
    y = np.array([1, 0, 1, 0])
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
    model.fit(X, y)

    class Wrapper:
        def __init__(self, m):
            self.model = m

        def predict(self, frame):
            return self.model.predict(frame)

    feature_matrix = pd.DataFrame({"zone_id": ["bk-tandon"], "f1": [1.0], "f2": [2.0]})
    rec = _score_with_learned_model(
        "bk-tandon", "Label", "subtype", feature_matrix, Wrapper(model), None
    )
    assert "f1" in rec.feature_contributions


def test_predict_cmf_sync_borough_fallback(monkeypatch) -> None:
    from src.api.routers.recommendations import predict_cmf_sync
    from src.schemas.requests import RecommendationRequest

    import src.api.routers.recommendations as rec_mod

    monkeypatch.setattr(rec_mod, "_SCORING_MODEL", None)
    monkeypatch.setattr(rec_mod, "_FEATURE_MATRIX", None)

    req = RecommendationRequest(concept_subtype="ramen", borough="XYZNOTREAL", limit=2)
    resp = predict_cmf_sync(req)
    assert len(resp.recommendations) > 0


def test_predict_cmf_sync_heuristic_path(monkeypatch) -> None:
    from src.api.routers.recommendations import predict_cmf_sync
    from src.schemas.requests import RecommendationRequest

    import src.api.routers.recommendations as rec_mod

    monkeypatch.setattr(rec_mod, "_SCORING_MODEL", None)
    monkeypatch.setattr(rec_mod, "_FEATURE_MATRIX", None)

    req = RecommendationRequest(concept_subtype="ramen", limit=1)
    resp = predict_cmf_sync(req)
    assert resp.recommendations[0].scoring_path == "heuristic"


def test_predict_cmf_sync_heuristic_fallback_mixed(monkeypatch) -> None:
    from src.api.routers.recommendations import predict_cmf_sync
    from src.schemas.requests import RecommendationRequest

    import src.api.routers.recommendations as rec_mod

    class DummyScorer:
        def predict(self, frame):
            return [0.5]

    feature_matrix = pd.DataFrame({"zone_id": ["bk-tandon"], "demand": [0.5]})

    monkeypatch.setattr(rec_mod, "_SCORING_MODEL", DummyScorer())
    monkeypatch.setattr(rec_mod, "_FEATURE_MATRIX", feature_matrix)

    req = RecommendationRequest(concept_subtype="ramen", limit=20)
    resp = predict_cmf_sync(req)
    paths = [r.scoring_path for r in resp.recommendations]
    assert "heuristic_fallback" in paths


@pytest.mark.asyncio
async def test_predict_trajectory_nonexistent_zone_type() -> None:
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/predict/trajectory", json={"zone_type": "ZZNONEXISTENT"}
        )
    assert resp.status_code == 200
    assert "trajectory_cluster" in resp.json()


def test_score_with_learned_model_survival_predict(monkeypatch) -> None:
    from src.api.routers.recommendations import _score_with_learned_model
    import numpy as np
    import pandas as pd

    class FakeScoring:
        def predict(self, X):
            return np.array([0.75])

        @property
        def feature_names(self):
            return ["feat1"]

    feature_matrix = pd.DataFrame(
        {"zone_id": ["Z1"], "feat1": [1.0], "time_key": [2024], "survival_score": [0.8]}
    )

    res = _score_with_learned_model(
        zone_id="Z1",
        zone_label="Zone 1",
        concept_subtype="healthy_indian",
        feature_matrix=feature_matrix,
        scoring_model=FakeScoring(),
        survival_model=None,
    )

    assert res is not None
    assert res.survival_risk == pytest.approx(0.2)  # 1.0 - 0.8


def test_score_with_learned_model_survival_no_score_defaults_to_half() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    class FakeScoring:
        def predict(self, X):
            return [0.5]

        @property
        def feature_names(self):
            return ["f1"]

    # No survival_score column → defaults to 0.5 viability → risk = 0.5
    feature_matrix = pd.DataFrame({"zone_id": ["Z1"], "f1": [1.0], "time_key": [2024]})
    res = _score_with_learned_model(
        "Z1", "L1", "S1", feature_matrix, FakeScoring(), None
    )
    assert res is not None
    assert res.survival_risk == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_lifespan_runs():
    from src.api.main import lifespan, app

    async with lifespan(app):
        pass


# ── _safe_float ───────────────────────────────────────────────────────────────


def test_safe_float_none_returns_fallback():
    from src.api.routers.recommendations import _safe_float

    assert _safe_float(None, 0.5) == 0.5


def test_safe_float_non_numeric_returns_fallback():
    from src.api.routers.recommendations import _safe_float

    assert _safe_float(object(), 0.5) == 0.5
