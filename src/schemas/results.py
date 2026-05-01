"""Outbound API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneRecommendation(BaseModel):
    """A detailed recommendation card."""

    zone_id: str
    zone_name: str
    rank: int = Field(default=0)
    score: float = Field(default=0.0)
    opportunity_score: float = Field(default=0.0)
    confidence_bucket: str = Field(default="low")
    concept_subtype: str
    positive_drivers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    positives: list[str] = Field(default_factory=list)
    similar_restaurants: list[str] = Field(default_factory=list)
    data_freshness: str = Field(default="2026-04")
    zone_type: str = Field(default="")
    borough: str = Field(default="Any")
    healthy_gap_summary: str = Field(default="")
    freshness_note: str = Field(default="No source refresh recorded yet.")
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    survival_risk: float = Field(default=0.0)
    model_version: str = Field(default="heuristic")
    scoring_path: str = Field(
        default="heuristic",
        description=(
            "Which scoring method produced this result: "
            "'learned', 'heuristic', 'heuristic_fallback'"
        ),
    )
    label_quality: float = Field(
        default=1.0,
        description="Fraction of ground truth components available for this zone (0-1)",
    )

    model_config = {"protected_namespaces": ()}


class RecommendationResponse(BaseModel):
    """Response payload for recommendation endpoints."""

    query: dict[str, str]
    recommendations: list[ZoneRecommendation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """System health and data stats."""

    status: str
    fm_row_count: int = 0
    feature_count: int = 0
    feature_matrix_rows: int = 0
    feature_matrix_cols: int = 0
    model_files_present: bool = False


def build_placeholder_response(
    concept_subtype: str, limit: int = 5
) -> RecommendationResponse:
    """Build deterministic placeholder cards for the frontend team."""

    recommendations = [
        ZoneRecommendation(
            zone_id=f"zone-{index}",
            zone_name=f"Placeholder Zone {index}",
            concept_subtype=concept_subtype,
            opportunity_score=round(0.75 - index * 0.05, 4),
            confidence_bucket="medium",
            healthy_gap_summary=(
                "Healthy options are under-supplied relative to quick lunch demand."
            ),
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
