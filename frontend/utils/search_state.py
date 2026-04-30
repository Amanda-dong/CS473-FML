from __future__ import annotations


def resolve_effective_search_settings(
    *,
    mode: str,
    has_description: bool,
    parsed_price_tier: str,
    parsed_risk_tolerance: str,
    selected_price_tier: str,
    selected_risk_tolerance: str,
    use_nlp_suggestions: bool,
) -> tuple[str, str]:
    """Return the effective price tier and risk tolerance for a submitted query."""
    if mode == "Describe my halal concept" and has_description and use_nlp_suggestions:
        return parsed_price_tier, parsed_risk_tolerance
    return selected_price_tier, selected_risk_tolerance
