"""Health endpoints for local development."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a minimal health payload for smoke tests."""

    return {"status": "ok"}
