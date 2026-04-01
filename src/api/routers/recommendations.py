"""Recommendation endpoints for the placeholder product."""

from fastapi import APIRouter

from src.schemas.requests import RecommendationRequest
from src.schemas.results import RecommendationResponse, build_placeholder_response
from src.utils.taxonomy import canonical_subtype

router = APIRouter(tags=["recommendations"])


@router.post("/predict/cmf", response_model=RecommendationResponse)
def predict_cmf(request: RecommendationRequest) -> RecommendationResponse:
    """Return placeholder recommendation cards for frontend wiring."""

    return build_placeholder_response(
        concept_subtype=canonical_subtype(request.concept_subtype),
        limit=request.limit,
    )


@router.post("/predict/trajectory")
def predict_trajectory(request: RecommendationRequest) -> dict[str, str]:
    """Return a placeholder macro neighborhood regime for the request."""

    return {
        "concept_subtype": canonical_subtype(request.concept_subtype),
        "trajectory_cluster": "emerging",
    }
