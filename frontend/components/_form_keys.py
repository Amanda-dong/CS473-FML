"""Single source of truth for sidebar control keys and defaults."""

from __future__ import annotations

FORM_KEYS: dict[str, str] = {
    "zone_type": "zone_type_select",
    "borough": "borough_select",
    "limit": "limit_slider",
    "concept": "concept_select",
    "custom_concept": "custom_concept_input",
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
    "price_tier": "mid",
    "risk_tolerance": "balanced",
    "compare_mode": False,
    "compare_concept": 1,
}
