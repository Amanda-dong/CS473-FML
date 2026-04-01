"""Helpers for resolving repository paths."""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings


def project_paths() -> dict[str, Path]:
    """Return the key directories used across the scaffold."""

    settings = get_settings()
    return {
        "repo_root": settings.repo_root,
        "data_dir": settings.data_dir,
        "raw_dir": settings.raw_dir,
        "processed_dir": settings.processed_dir,
        "geojson_dir": settings.geojson_dir,
    }
