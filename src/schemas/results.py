"""Outbound API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneRecommendation(BaseModel):
    """A single placeholder recommendation card."""

    zone_id: str
    zone_name: str
    concept_subtype: str
    opportunity_score: float = Field(default=0.0)
    confidence_bucket: str = Field(default="low")
    healthy_gap_summary: str = Field(default="")
    positives: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    freshness_note: str = Field(default="No source refresh recorded yet.")


class RecommendationResponse(BaseModel):
    """Response payload for recommendation endpoints."""

    query: dict[str, str]
    recommendations: list[ZoneRecommendation] = Field(default_factory=list)


def build_placeholder_response(concept_subtype: str, limit: int = 5) -> RecommendationResponse:
    """Build deterministic placeholder cards for the frontend team."""

    recommendations = [
        ZoneRecommendation(
            zone_id=f"zone-{index}",
            zone_name=f"Placeholder Zone {index}",
            concept_subtype=concept_subtype,
            opportunity_score=80.0 - index,
            confidence_bucket="medium",
            healthy_gap_summary="Healthy options are under-supplied relative to quick lunch demand.",
            positives=[
                "Strong daytime footfall proxy",
                "Low healthy subtype saturation",
                "Good fit for lunch-oriented service",
            ],
            risks=[
                "Rent pressure not audited yet",
                "Yelp coverage still needs validation",
            ],
        )
        for index in range(1, limit + 1)
    ]
    return RecommendationResponse(
        query={"concept_subtype": concept_subtype},
        recommendations=recommendations,
    )
