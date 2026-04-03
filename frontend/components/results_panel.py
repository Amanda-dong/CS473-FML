"""Results panel for recommendation cards and diagnostics."""

from __future__ import annotations

import streamlit as st

from frontend.components.map_view import render_map_view
from frontend.components.recommendation_card import render_recommendation_card
from src.schemas.results import build_placeholder_response
from src.utils.taxonomy import canonical_subtype


def render_results_panel(user_state: dict[str, str]) -> None:
    """Render recommendations driven by the current controls.

    Attempts to call the live API at http://localhost:8000/predict/cmf.
    Falls back to build_placeholder_response() on connection failure.
    """

    concept_subtype = canonical_subtype(user_state.get("concept_subtype", "healthy_indian"))
    limit = int(user_state.get("limit", 5))
    using_placeholder = False

    recommendations = []
    # Try live API first; fall back to in-process scoring pipeline
    try:
        import httpx

        payload = {k: v for k, v in user_state.items() if v is not None}
        payload.setdefault("concept_subtype", concept_subtype)
        payload.setdefault("limit", limit)
        resp = httpx.post("http://localhost:8000/predict/cmf", json=payload, timeout=3.0)
        resp.raise_for_status()
        recommendations = resp.json().get("recommendations", [])
    except Exception:
        # In-process fallback — runs the real scoring without a separate server
        try:
            from src.api.routers.recommendations import predict_cmf
            from src.schemas.requests import RecommendationRequest

            req = RecommendationRequest(
                concept_subtype=concept_subtype,
                price_tier=str(user_state.get("price_tier", "mid")),
                borough=user_state.get("borough") or None,
                risk_tolerance=str(user_state.get("risk_tolerance", "balanced")),
                zone_type=str(user_state.get("zone_type", "campus_walkshed")),
                limit=limit,
            )
            response = predict_cmf(req)
            recommendations = [card.model_dump() for card in response.recommendations]
        except Exception:
            using_placeholder = True
            response = build_placeholder_response(concept_subtype=concept_subtype, limit=limit)
            recommendations = [card.model_dump() for card in response.recommendations]

    if using_placeholder:
        st.info(
            "Live API unavailable — showing placeholder recommendations. "
            "Start the backend with `uv run python -m uvicorn src.api.main:app` to load real scores."
        )

    st.subheader("Recommended Zones")
    for card in recommendations:
        if not isinstance(card, dict):
            card = dict(card)
        render_recommendation_card(card)

    render_map_view()
