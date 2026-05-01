"""Scenario controls — supports any cuisine type via free-text or dropdown."""

from __future__ import annotations

import streamlit as st

from src.utils.taxonomy import all_known_subtypes, canonical_subtype

from frontend.components._form_keys import FORM_KEYS

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

_CONCEPT_DESCRIPTIONS: dict[str, str] = {
    "healthy_indian": "South Asian cuisine with modern healthy fast-casual positioning — think tandoor bowls, daal, and grilled proteins over rice.",
    "mediterranean_bowls": "Mediterranean grain bowls, mezze plates, and falafel wraps — high overlap with salad-forward and bowl-format dining.",
    "salad_bowls": "Salad-forward fast-casual (Sweetgreen-style) — customizable bases, toppings, and dressings with quick turnaround.",
    "vegan_grab_and_go": "Explicitly plant-based or vegetarian quick-service — cold-pressed juices, wraps, and grab-and-go snack items.",
    "protein_forward_lunch": "High-protein lunch formats — grilled chicken, steak bowls, or macro-focused fast-casual for fitness-adjacent demand.",
    "smoothie_juice": "Smoothie bars and cold-pressed juice concepts — high ticket, low footprint, strong campus and gym-adjacent demand.",
    "bakery_cafe": "Café and bakery format — morning peak + remote-worker daytime dwell time; low healthy-food competition signal.",
}


def render_scenario_panel() -> dict[str, str | bool | None]:
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
        key=FORM_KEYS["concept"],
        help="Choose the healthy-food concept you want to locate. Use 'Custom' to enter any cuisine.",
    )
    selected = subtypes[selected_idx]  # type: ignore[index]

    if selected in _CONCEPT_DESCRIPTIONS:
        st.caption(_CONCEPT_DESCRIPTIONS[selected])

    if selected == "__custom__":
        custom = st.text_input(
            "Enter your concept (e.g. 'healthy Korean', 'Peruvian ceviche', 'bubble tea')",
            placeholder="Any cuisine or concept...",
            key=FORM_KEYS["custom_concept"],
        )
        concept_subtype = canonical_subtype(custom) if custom.strip() else "unknown"
    else:
        concept_subtype = selected

    price_tier = st.selectbox(
        "Price tier",
        ["budget", "mid", "premium"],
        key=FORM_KEYS["price_tier"],
    )
    risk_tolerance = st.selectbox(
        "Risk tolerance",
        ["conservative", "balanced", "aggressive"],
        key=FORM_KEYS["risk_tolerance"],
    )

    compare_mode = st.checkbox(
        "Compare two concepts",
        value=False,
        key=FORM_KEYS["compare_mode"],
        help="Score a second concept side-by-side to contrast opportunity zones.",
    )
    compare_concept: str | None = None
    if compare_mode:
        # Drop __custom__ from compare list for simplicity
        compare_subtypes = list(all_known_subtypes())
        compare_labels = [
            _DISPLAY_NAMES.get(s, s.replace("_", " ").title()) for s in compare_subtypes
        ]
        default_idx = 1 if len(compare_subtypes) > 1 else 0
        compare_idx = st.selectbox(
            "Compare with",
            options=range(len(compare_subtypes)),
            format_func=lambda i: compare_labels[i],
            index=default_idx,
            key=FORM_KEYS["compare_concept"],
        )
        compare_concept = compare_subtypes[compare_idx]

    return {
        "concept_subtype": concept_subtype,
        "price_tier": price_tier,
        "risk_tolerance": risk_tolerance,
        "compare_mode": compare_mode,
        "compare_concept": compare_concept,
    }
