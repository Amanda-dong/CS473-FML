"""Inbound API request schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecommendationRequest(BaseModel):
    """User input for healthy-food site recommendation."""

    model_config = ConfigDict(populate_by_name=True)

    concept_subtype: str = Field(default="healthy_indian")
    price_tier: Literal["budget", "mid", "premium"] = Field(default="mid")
    borough: Optional[str] = None
    risk_tolerance: str = Field(default="balanced")
    zone_type: str = Field(default="")
    max_results: int = Field(default=5, ge=1, le=20, alias="limit")

    @field_validator("concept_subtype")
    @classmethod
    def validate_subtype(cls, v: str) -> str:
        """Reject non-alphanumeric (except spaces/underscores) to prevent injection."""
        if not v:
            raise ValueError("concept_subtype must not be empty")
        # Allow alphanumeric, spaces, and underscores only
        if not all(c.isalnum() or c in " _" for c in v):
            raise ValueError(
                "concept_subtype must be alphanumeric, spaces, or underscores only"
            )
        return v
