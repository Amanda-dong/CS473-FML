"""Tests for the API scaffold."""

from src.api.main import app


def test_api_title() -> None:
    """The FastAPI app should expose the intended title."""

    assert app.title == "NYC Restaurant Intelligence Platform API"


def test_api_has_routes() -> None:
    """The scaffold should register core route handlers."""

    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths
    assert "/datasets" in route_paths
    assert "/predict/cmf" in route_paths
