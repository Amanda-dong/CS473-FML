"""Single source of truth for sidebar control keys and defaults."""

from __future__ import annotations

FORM_KEYS: dict[str, str] = {
    "zone_type": "zone_type_select",
    "borough": "borough_select",
    "limit": "limit_slider",
    "concept_mode": "concept_mode_radio",
    "concept": "concept_select",
    "custom_concept": "custom_concept_input",
    "nlp_concept": "nlp_concept_input",
    "use_nlp_suggestions": "use_nlp_suggestions_cb",
    "price_tier": "price_tier_select",
    "risk_tolerance": "risk_tolerance_select",
    "compare_mode": "compare_mode_cb",
    "compare_concept": "compare_concept_select",
}

FORM_DEFAULTS: dict[str, object] = {
    "zone_type": "All",
    "borough": "Any",
    "limit": 5,
    "concept": 0,
    "custom_concept": "",
    "concept_mode": "Describe my halal concept",
    "nlp_concept": "",
    "use_nlp_suggestions": True,
    "price_tier": "mid",
    "risk_tolerance": "balanced",
    "compare_mode": False,
    "compare_concept": 1,
}
