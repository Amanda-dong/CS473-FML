"""Taxonomy helpers for healthy-food concepts and local competition."""

from __future__ import annotations

from src.config import HEALTHY_SUBTYPES

_KEYWORDS = {
    "salad_bowls": ("salad", "greens", "bowl"),
    "mediterranean_bowls": ("mediterranean", "cava", "naya", "grain bowl"),
    "healthy_indian": ("indian", "south asian", "chaat", "tandoor", "protein bowl"),
    "vegan_grab_and_go": ("vegan", "plant-based", "vegetarian"),
    "protein_forward_lunch": ("protein", "lean", "high-protein"),
}


def healthy_taxonomy() -> dict[str, tuple[str, ...]]:
    """Return the placeholder taxonomy used in demand and gap features."""

    return dict(_KEYWORDS)


def canonical_subtype(raw_value: str) -> str:
    """Normalize user input to a known concept subtype when possible."""

    normalized = raw_value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in HEALTHY_SUBTYPES:
        return normalized
    for subtype, keywords in _KEYWORDS.items():
        if any(keyword.replace(" ", "_") in normalized for keyword in keywords):
            return subtype
    return normalized
