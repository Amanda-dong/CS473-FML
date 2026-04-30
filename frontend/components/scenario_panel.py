"""Scenario controls for halal-focused merchants using structured or text input."""

from __future__ import annotations

import re

import streamlit as st

from src.utils.taxonomy import all_known_subtypes, canonical_subtype

from frontend.components._form_keys import FORM_KEYS
from frontend.utils.search_state import resolve_effective_search_settings

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
    "halal": "Halal-first fast casual, carts, bowls, wraps, grills, or comfort food anchored in halal sourcing and Muslim-friendly positioning.",
    "healthy_indian": "South Asian cuisine with modern healthy fast-casual positioning — think tandoor bowls, daal, and grilled proteins over rice.",
    "mediterranean_bowls": "Mediterranean grain bowls, mezze plates, and falafel wraps — high overlap with salad-forward and bowl-format dining.",
    "salad_bowls": "Salad-forward fast-casual (Sweetgreen-style) — customizable bases, toppings, and dressings with quick turnaround.",
    "vegan_grab_and_go": "Explicitly plant-based or vegetarian quick-service — cold-pressed juices, wraps, and grab-and-go snack items.",
    "protein_forward_lunch": "High-protein lunch formats — grilled chicken, steak bowls, or macro-focused fast-casual for fitness-adjacent demand.",
    "smoothie_juice": "Smoothie bars and cold-pressed juice concepts — high ticket, low footprint, strong campus and gym-adjacent demand.",
    "bakery_cafe": "Café and bakery format — morning peak + remote-worker daytime dwell time; low healthy-food competition signal.",
}

_MODE_HELP = {
    "Describe my halal concept": "Describe your idea in plain language.",
    "Use structured controls": "Pick a concept from the list.",
}

_PRICE_KEYWORDS = {
    "budget": (
        "budget",
        "affordable",
        "cheap",
        "value",
        "student-friendly",
        "low price",
        "quick bite",
    ),
    "premium": (
        "premium",
        "upscale",
        "fine dining",
        "higher-end",
        "luxury",
        "elevated",
    ),
}

_RISK_KEYWORDS = {
    "conservative": (
        "low risk",
        "safe",
        "proven",
        "steady",
        "family-friendly",
        "stable",
    ),
    "aggressive": (
        "bold",
        "aggressive",
        "experimental",
        "late-night",
        "trend-driven",
        "high upside",
    ),
}

_SERVICE_KEYWORDS = {
    "fast casual": ("fast casual", "quick service", "counter service", "grab-and-go"),
    "sit-down": ("sit-down", "full service", "table service", "dining room"),
    "late-night": ("late-night", "after midnight", "open late", "night crowd"),
    "lunch-led": ("lunch", "office workers", "students", "quick lunch"),
}

_AUDIENCE_KEYWORDS = {
    "students": ("students", "campus", "college", "university"),
    "office workers": ("office", "commuters", "midtown", "workers"),
    "families": ("families", "kids", "parents", "neighborhood"),
    "gym and wellness": ("gym", "fitness", "healthy", "protein"),
}


def _match_subtype_index(subtypes: list[str], target: str) -> int:
    try:
        return subtypes.index(target)
    except ValueError:
        return 0


def _render_structured_concept_picker(subtypes: list[str]) -> str:
    display_labels = [
        _DISPLAY_NAMES.get(s, s.replace("_", " ").title()) for s in subtypes
    ]
    default_idx = _match_subtype_index(subtypes, "halal")
    selected_idx = st.selectbox(
        "Halal restaurant style",
        options=range(len(subtypes)),
        format_func=lambda i: display_labels[i],
        index=default_idx,
        key=FORM_KEYS["concept"],
        help="Choose the kind of halal food concept you want to test first.",
    )
    selected = subtypes[selected_idx]  # type: ignore[index]
    if selected in _CONCEPT_DESCRIPTIONS:
        st.caption(_CONCEPT_DESCRIPTIONS[selected])

    if selected == "__custom__":
        custom = st.text_input(
            "Enter a custom halal concept",
            placeholder="Examples: halal smash burgers, halal Korean bowls, late-night halal grill...",
            key=FORM_KEYS["custom_concept"],
        )
        return canonical_subtype(custom) if custom.strip() else "halal"
    return selected


def _extract_keyword_match(
    prompt: str, keyword_map: dict[str, tuple[str, ...]], fallback: str
) -> str:
    lowered = prompt.lower()
    for label, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return label
    return fallback


def _extract_multi_matches(
    prompt: str, keyword_map: dict[str, tuple[str, ...]]
) -> list[str]:
    lowered = prompt.lower()
    matches = [
        label
        for label, keywords in keyword_map.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return matches


def _looks_halal_focused(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        token in lowered
        for token in (
            "halal",
            "muslim",
            "zabiha",
            "zabihah",
        )
    )


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip()


def _render_nlp_concept_input() -> tuple[str, str, str, str]:
    user_prompt = st.text_area(
        "Describe the halal restaurant you want to open",
        key=FORM_KEYS["nlp_concept"],
        height=140,
        placeholder=(
            "Example: I want to open a halal fast-casual lunch spot with grilled chicken, "
            "rice bowls, fresh sides, and quick service for students and office workers."
        ),
        help="We will map your description to a concept and suggest pricing and risk settings.",
    )
    cleaned_prompt = _normalize_prompt(user_prompt)
    normalized = canonical_subtype(cleaned_prompt) if cleaned_prompt else "halal"
    if not cleaned_prompt:
        st.info("Describe your halal restaurant idea to get a suggested match.")
        return normalized, cleaned_prompt, "mid", "balanced"
    label = _DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())
    suggested_price = _extract_keyword_match(cleaned_prompt, _PRICE_KEYWORDS, "mid")
    suggested_risk = _extract_keyword_match(cleaned_prompt, _RISK_KEYWORDS, "balanced")
    service_matches = _extract_multi_matches(cleaned_prompt, _SERVICE_KEYWORDS)
    audience_matches = _extract_multi_matches(cleaned_prompt, _AUDIENCE_KEYWORDS)

    st.caption("Review the parsed summary below. You can still change price and risk.")
    if cleaned_prompt and not _looks_halal_focused(cleaned_prompt):
        st.warning(
            "Your prompt does not mention halal. Add halal-specific wording for a better match."
        )

    parsed_c1, parsed_c2 = st.columns(2)
    with parsed_c1:
        st.markdown(f"**Parsed concept:** {label}")
        st.markdown(f"**Suggested price tier:** {suggested_price.title()}")
    with parsed_c2:
        st.markdown(f"**Suggested risk tolerance:** {suggested_risk.title()}")
        st.markdown(
            f"**Detected service style:** {', '.join(service_matches) if service_matches else 'Not clearly specified'}"
        )
    st.caption(
        "Audience: "
        + (", ".join(audience_matches) if audience_matches else "Not specified")
    )
    return normalized, cleaned_prompt, suggested_price, suggested_risk


def render_scenario_panel() -> dict[str, str | bool | None]:
    """Render concept, price, and risk controls with halal-first messaging."""
    subtypes = list(all_known_subtypes()) + ["__custom__"]
    mode = st.radio(
        "How would you like to define your halal concept?",
        options=list(_MODE_HELP.keys()),
        key=FORM_KEYS["concept_mode"],
        help="Choose between structured dropdowns and an NLP-style text description.",
    )
    st.caption(_MODE_HELP[mode])

    user_concept_text: str | None = None
    parsed_price_tier = "mid"
    parsed_risk_tolerance = "balanced"
    compare_mode = False
    compare_concept: str | None = None

    if mode == "Use structured controls":
        concept_subtype = _render_structured_concept_picker(subtypes)
        price_tier = st.selectbox(
            "Price tier",
            ["budget", "mid", "premium"],
            key=FORM_KEYS["price_tier"],
            help="Tell the model whether your halal concept is meant to be value-oriented, mainstream mid-market, or premium.",
        )
        risk_tolerance = st.selectbox(
            "Risk tolerance",
            ["conservative", "balanced", "aggressive"],
            key=FORM_KEYS["risk_tolerance"],
            help="Choose how adventurous you want the shortlist to be for a new halal opening.",
        )
        compare_mode = st.checkbox(
            "Compare two concepts",
            value=False,
            key=FORM_KEYS["compare_mode"],
            help="Score a second concept side-by-side to contrast opportunity zones.",
        )
        if compare_mode:
            compare_subtypes = list(all_known_subtypes())
            compare_labels = [
                _DISPLAY_NAMES.get(s, s.replace("_", " ").title())
                for s in compare_subtypes
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
        use_nlp_suggestions = False
    else:
        concept_subtype, user_concept_text, parsed_price_tier, parsed_risk_tolerance = (
            _render_nlp_concept_input()
        )
        price_tier = parsed_price_tier
        risk_tolerance = parsed_risk_tolerance
        use_nlp_suggestions = True

    effective_price_tier, effective_risk_tolerance = resolve_effective_search_settings(
        mode=mode,
        has_description=bool(user_concept_text),
        parsed_price_tier=parsed_price_tier,
        parsed_risk_tolerance=parsed_risk_tolerance,
        selected_price_tier=price_tier,
        selected_risk_tolerance=risk_tolerance,
        use_nlp_suggestions=use_nlp_suggestions,
    )

    return {
        "concept_mode": mode,
        "concept_subtype": concept_subtype,
        "concept_description": user_concept_text,
        "parsed_price_tier": parsed_price_tier,
        "parsed_risk_tolerance": parsed_risk_tolerance,
        "use_nlp_suggestions": use_nlp_suggestions,
        "price_tier": effective_price_tier,
        "risk_tolerance": effective_risk_tolerance,
        "compare_mode": compare_mode,
        "compare_concept": compare_concept,
    }
