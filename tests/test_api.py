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
