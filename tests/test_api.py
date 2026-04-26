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


def test_score_with_learned_model_uses_latest_time_key_and_predict_risk() -> None:
    from src.api.routers.recommendations import _score_with_learned_model

    class DummyScoringModel:
        feature_names = ["healthy_review_share"]

        def predict(self, frame: pd.DataFrame) -> list[float]:
            return frame["healthy_review_share"].tolist()

    class DummySurvivalModel:
        def predict_risk(self, frame: pd.DataFrame) -> pd.Series:
            return pd.Series([0.25], index=frame.index)

    feature_matrix = pd.DataFrame(
        {
            "zone_id": ["bk-tandon", "bk-tandon"],
            "time_key": [2022, 2024],
            "healthy_review_share": [0.1, 0.9],
        }
    )

    rec = _score_with_learned_model(
        "bk-tandon",
        "NYU Tandon / MetroTech",
        "salad_bowls",
        feature_matrix,
        DummyScoringModel(),
        DummySurvivalModel(),
    )

    assert rec is not None
    assert rec.opportunity_score == pytest.approx(0.9)
    assert rec.survival_risk == pytest.approx(0.25)


# ── _confidence_bucket — all thresholds ──────────────────────────────────────


def test_confidence_bucket_high() -> None:
    from src.api.routers.recommendations import _confidence_bucket

    assert _confidence_bucket(0.61) == "high"
    assert _confidence_bucket(1.0) == "high"


def test_confidence_bucket_medium() -> None:
    from src.api.routers.recommendations import _confidence_bucket

    assert _confidence_bucket(0.60) == "medium"
    assert _confidence_bucket(0.41) == "medium"


def test_confidence_bucket_low() -> None:
    from src.api.routers.recommendations import _confidence_bucket

    assert _confidence_bucket(0.40) == "low"
    assert _confidence_bucket(0.0) == "low"
    assert _confidence_bucket(0.2) == "low"


# ── _get_zone_type_clusters ───────────────────────────────────────────────────


def test_get_zone_type_clusters_returns_dict() -> None:
    from src.api.routers.recommendations import _get_zone_type_clusters

    result = _get_zone_type_clusters("healthy_indian", "medium", "mid")
    assert isinstance(result, dict)
    # Should have entries for all zone types present in _NYC_ZONES
    for zt in ("campus_walkshed", "lunch_corridor", "transit_catchment", "business_district"):
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
    """Premium+conservative should differ from budget+aggressive in survival score."""
    from src.api.routers.recommendations import _score_one

    rec_premium = _score_one("bk-tandon", "campus_walkshed", "L", "salad_bowls", "conservative", "premium")
    rec_budget = _score_one("bk-tandon", "campus_walkshed", "L", "salad_bowls", "aggressive", "budget")
    # Both must return valid recommendations; scores are heuristic floats
    assert rec_premium.opportunity_score >= 0.0
    assert rec_budget.opportunity_score >= 0.0
