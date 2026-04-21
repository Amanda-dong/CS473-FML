"""Scenario controls — supports any cuisine type via free-text or dropdown."""

from __future__ import annotations

import streamlit as st

from src.utils.taxonomy import all_known_subtypes, canonical_subtype

_DISPLAY_NAMES: dict[str, str] = {
    "healthy_indian": "Healthy Indian / South Asian",
    "mediterranean_bowls": "Mediterranean Bowls",
    "salad_bowls": "Salad Bowls",
    "vegan_grab_and_go": "Vegan / Plant-Based",
    "protein_forward_lunch": "Protein-Forward Lunch",
    "ramen": "Ramen",
    "dim_sum": "Dim Sum",
    "japanese": "Japanese",
    "korean": "Korean / K-BBQ",
    "chinese": "Chinese",
    "thai": "Thai",
    "mexican": "Mexican / Tacos",
    "caribbean": "Caribbean",
    "ethiopian": "Ethiopian",
    "west_african": "West African",
    "middle_eastern": "Middle Eastern",
    "greek": "Greek",
    "italian": "Italian",
    "pizza": "Pizza",
    "american_comfort": "American Comfort / BBQ",
    "burgers": "Burgers",
    "seafood": "Seafood",
    "bakery_cafe": "Bakery / Café",
    "smoothie_juice": "Smoothies & Juice Bar",
    "__custom__": "Custom — type below...",
}


def render_scenario_panel() -> dict[str, str]:
    """Render concept, price, and risk controls.  Supports any cuisine type."""
    subtypes = list(all_known_subtypes()) + ["__custom__"]
    display_labels = [
        _DISPLAY_NAMES.get(s, s.replace("_", " ").title()) for s in subtypes
    ]

    selected_idx = st.selectbox(
        "Cuisine / concept type",
        options=range(len(subtypes)),
        format_func=lambda i: display_labels[i],
        index=0,
    )
    selected = subtypes[selected_idx]  # type: ignore[index]

    if selected == "__custom__":
        custom = st.text_input(
            "Enter your concept (e.g. 'healthy Korean', 'Peruvian ceviche', 'bubble tea')",
            placeholder="Any cuisine or concept...",
        )
        concept_subtype = canonical_subtype(custom) if custom.strip() else "unknown"
    else:
        concept_subtype = selected

    price_tier = st.selectbox("Price tier", ["budget", "mid", "premium"])
    risk_tolerance = st.selectbox(
        "Risk tolerance", ["conservative", "balanced", "aggressive"]
    )

    return {
        "concept_subtype": concept_subtype,
        "price_tier": price_tier,
        "risk_tolerance": risk_tolerance,
    }
