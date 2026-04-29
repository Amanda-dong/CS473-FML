"""Inbound API request schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """User input for healthy-food site recommendation."""

    concept_subtype: str = Field(default="healthy_indian")
    price_tier: str = Field(default="mid")
    borough: Optional[str] = None
    risk_tolerance: str = Field(default="balanced")
    zone_type: str = Field(default="")
    limit: int = Field(default=5, ge=1, le=50)
