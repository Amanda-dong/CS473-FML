"""Application settings for local development."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """File-system and project defaults used across the scaffold."""

    project_name: str = "NYC Restaurant Intelligence Platform"
    target_use_case: str = "healthy-food white-space recommendations"
    default_city: str = "New York City"
    default_time_grain: str = "year"
    microzone_minutes: int = 10
    repo_root: Path = Path(__file__).resolve().parents[2]

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def geojson_dir(self) -> Path:
        return self.data_dir / "geojson"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object for the current process."""

    return Settings()
