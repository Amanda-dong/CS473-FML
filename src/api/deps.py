"""Shared API dependencies."""

from src.config import Settings, get_settings


def get_app_settings() -> Settings:
    """Expose cached settings via FastAPI dependency injection."""

    return get_settings()
