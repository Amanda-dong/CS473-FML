"""Results panel for recommendation cards and diagnostics."""

import streamlit as st

from frontend.components.map_view import render_map_view
from frontend.components.recommendation_card import render_recommendation_card
from src.schemas.results import build_placeholder_response
from src.utils.taxonomy import canonical_subtype


def render_results_panel(user_state: dict[str, str]) -> None:
    """Render placeholder recommendations driven by the current controls."""

    concept_subtype = canonical_subtype(user_state.get("concept_subtype", "healthy_indian"))
    response = build_placeholder_response(concept_subtype=concept_subtype, limit=3)
    st.subheader("Recommended Zones")
    for card in response.recommendations:
        render_recommendation_card(card.model_dump())
    render_map_view()
