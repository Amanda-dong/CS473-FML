"""Tests for frontend query resolution logic."""

from frontend.utils.search_state import resolve_effective_search_settings


def test_resolve_effective_search_settings_uses_nlp_values_when_enabled() -> None:
    price, risk = resolve_effective_search_settings(
        mode="Describe my halal concept",
        has_description=True,
        parsed_price_tier="premium",
        parsed_risk_tolerance="aggressive",
        selected_price_tier="mid",
        selected_risk_tolerance="balanced",
        use_nlp_suggestions=True,
    )
    assert price == "premium"
    assert risk == "aggressive"


def test_resolve_effective_search_settings_uses_selected_values_when_structured() -> (
    None
):
    price, risk = resolve_effective_search_settings(
        mode="Use structured controls",
        has_description=False,
        parsed_price_tier="premium",
        parsed_risk_tolerance="aggressive",
        selected_price_tier="budget",
        selected_risk_tolerance="conservative",
        use_nlp_suggestions=False,
    )
    assert price == "budget"
    assert risk == "conservative"


def test_resolve_effective_search_settings_falls_back_without_description() -> None:
    price, risk = resolve_effective_search_settings(
        mode="Describe my halal concept",
        has_description=False,
        parsed_price_tier="premium",
        parsed_risk_tolerance="aggressive",
        selected_price_tier="mid",
        selected_risk_tolerance="balanced",
        use_nlp_suggestions=True,
    )
    assert price == "mid"
    assert risk == "balanced"
