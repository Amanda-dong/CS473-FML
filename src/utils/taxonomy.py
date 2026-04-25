"""Taxonomy helpers for any cuisine concept and local competition."""

from __future__ import annotations

from src.config import HEALTHY_SUBTYPES

# ── Keyword map ───────────────────────────────────────────────────────────────
# More-specific subtypes MUST appear before broader catch-all terms so the
# classifier matches the correct bucket on the first keyword hit.
# Keys match HEALTHY_SUBTYPES entries; free-text entries pass through as-is.

_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Healthy / wellness
    "healthy_indian": ("healthy indian", "south asian bowl", "chaat", "tandoor"),
    "mediterranean_bowls": (
        "mediterranean",
        "cava",
        "naya",
        "grain bowl",
        "falafel bowl",
        "shawarma bowl",
    ),
    "vegan_grab_and_go": ("vegan", "plant-based", "vegetarian grab"),
    "protein_forward_lunch": (
        "high-protein",
        "protein-forward",
        "lean lunch",
        "macro bowl",
    ),
    "salad_bowls": ("salad bowl", "salad bar", "greens"),
    # Global cuisines
    "ramen": ("ramen", "tonkotsu", "shoyu ramen", "miso ramen"),
    "dim_sum": ("dim sum", "dimsum", "har gow", "siu mai", "yum cha"),
    "japanese": (
        "japanese",
        "sushi",
        "sashimi",
        "izakaya",
        "teriyaki",
        "udon",
        "tempura",
    ),
    "korean": ("korean", "bibimbap", "bulgogi", "kimchi", "korean bbq", "kbbq"),
    "chinese": ("chinese", "cantonese", "szechuan", "peking", "xiao long bao"),
    "thai": ("thai", "pad thai", "green curry", "thai basil"),
    "mexican": (
        "mexican",
        "taco",
        "burrito",
        "torta",
        "birria",
        "tlayuda",
        "enchilada",
    ),
    "caribbean": ("caribbean", "jamaican", "jerk", "plantain", "trinidadian"),
    "ethiopian": ("ethiopian", "injera", "tibs", "kitfo"),
    "west_african": ("west african", "nigerian", "ghanaian", "senegalese", "jollof"),
    "middle_eastern": (
        "middle eastern",
        "lebanese",
        "hummus",
        "mezze",
        "persian",
        "turkish",
    ),
    "greek": ("greek", "gyro", "souvlaki", "spanakopita"),
    "italian": ("italian", "pasta", "risotto", "osteria", "trattoria", "carbonara"),
    "pizza": ("pizza", "pizzeria", "neapolitan pizza", "calzone"),
    "american_comfort": (
        "american comfort",
        "diner",
        "bbq",
        "fried chicken",
        "mac and cheese",
    ),
    "burgers": ("burger", "smash burger", "patty", "cheeseburger"),
    "seafood": ("seafood", "oyster", "lobster", "crab", "fish and chips", "poke"),
    "bakery_cafe": ("bakery", "pastry", "croissant", "coffee shop", "café", "cafe"),
    "smoothie_juice": ("smoothie", "juice bar", "acai", "blend"),
}


def healthy_taxonomy() -> dict[str, tuple[str, ...]]:
    """Return the full cuisine taxonomy used in demand and gap features."""
    return dict(_KEYWORDS)


def canonical_subtype(raw_value: str) -> str:
    """Normalize user input to a known concept subtype when possible.

    Unknown values are slug-ified and returned as-is so the system remains
    open to any cuisine type rather than rejecting unfamiliar inputs.
    """
    normalized = raw_value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in HEALTHY_SUBTYPES:
        return normalized
    # Try keyword matching in the raw (lowered) text
    lowered = raw_value.strip().lower()
    for subtype, keywords in _KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return subtype
    # Return slug of whatever the user typed — custom subtype
    return normalized


def all_known_subtypes() -> tuple[str, ...]:
    """Return every named subtype understood by the taxonomy."""
    return HEALTHY_SUBTYPES
