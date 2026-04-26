"""Schemas for dataset metadata and audits."""

from typing import Optional

from pydantic import BaseModel, Field


class DatasetAuditRow(BaseModel):
    """Metadata that lets the team audit a source before using it."""

    name: str
    owner: str
    spatial_unit: str
    time_grain: str
    earliest_year: Optional[int] = None
    status: str = Field(default="planned")
    notes: str = Field(default="")
