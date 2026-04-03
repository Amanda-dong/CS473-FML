"""Turn zone feature dicts into human-readable recommendation-card explanations."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def top_positive_drivers(zone_features: Mapping[str, float]) -> list[str]:
    """Return quantitative explanation strings for the strongest positive signals."""
    drivers: list[str] = []

    quick_lunch = zone_features.get("quick_lunch_demand", 0.0)
    if quick_lunch > 0.6:
        drivers.append(f"High daytime foot-traffic / lunch-demand index ({quick_lunch:.0%})")

    subtype_gap = zone_features.get("subtype_gap", 0.0)
    if subtype_gap > 0.5:
        drivers.append(
            f"Strong cuisine gap ({subtype_gap:.0%}) — this concept is under-supplied here"
        )

    survival_score = zone_features.get("survival_score", 0.0)
    if survival_score > 0.6:
        drivers.append(f"Survival model gives {survival_score:.0%} commercial viability")

    license_velocity = zone_features.get("license_velocity", 0.0)
    if license_velocity > 0.3:
        drivers.append("Positive license velocity — active neighborhood growth signal")

    healthy_review_share = zone_features.get("healthy_review_share", 0.0)
    if healthy_review_share > 0.3:
        drivers.append(f"NLP review signals show {healthy_review_share:.0%} demand for this category")

    transit_access = zone_features.get("transit_access", 0.0)
    if transit_access > 0.75:
        drivers.append(f"Excellent transit accessibility ({transit_access:.0%}) — maximises foot-traffic")

    income_alignment = zone_features.get("income_alignment", 0.0)
    if income_alignment > 0.65:
        drivers.append("Income profile aligns well with the chosen price tier")

    return drivers or ["Explanation rules not configured yet"]


def top_risks(zone_features: Mapping[str, float]) -> list[str]:
    """Return quantitative risk strings for recommendation cards."""
    risks: list[str] = []

    rent_pressure = zone_features.get("rent_pressure", 0.0)
    if rent_pressure > 0.5:
        risks.append(f"High rent pressure ({rent_pressure:.0%}) — may compress margins significantly")

    competition_score = zone_features.get("competition_score", 0.0)
    if competition_score > 0.5:
        risks.append(
            f"Saturated market ({competition_score:.0%} competitor density) — differentiation required"
        )

    survival_score = zone_features.get("survival_score", 0.0)
    if survival_score < 0.4:
        risks.append("Below-average survival outlook — consider more established zone")

    income_alignment = zone_features.get("income_alignment", 0.0)
    if income_alignment < 0.35:
        risks.append("Income/price-tier mismatch — local spending power may not support this concept")

    transit_access = zone_features.get("transit_access", 0.0)
    if transit_access < 0.45:
        risks.append("Limited transit access — foot-traffic relies on local residents only")

    return risks or ["Risk rules not configured yet"]


# ---------------------------------------------------------------------------
# SHAP-based explainability (Phase 4)
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "demand_signal": "Daytime foot-traffic demand",
    "subtype_gap": "Cuisine white-space opportunity",
    "survival_score": "Commercial viability",
    "rent_pressure": "Rent pressure",
    "competition_score": "Market competition",
    "license_velocity": "Neighborhood growth signal",
    "healthy_review_share": "NLP demand confirmation",
    "transit_access": "Transit accessibility",
    "income_alignment": "Price-tier income fit",
    "inspection_grade_avg": "Health inspection quality",
    "healthy_gap_score": "Category under-supply",
    "quick_lunch_demand": "Lunch-traffic demand index",
    "review_demand_score": "Review-based demand signal",
    "merchant_viability_score": "Merchant viability outlook",
}


def shap_drivers(model: object, X_row: pd.Series, top_n: int = 3) -> tuple[list[str], list[str]]:
    """SHAP-based top positive and negative drivers for a single prediction.

    Parameters
    ----------
    model : LearnedScoringModel
        A fitted model with an ``explain()`` method.
    X_row : pd.Series
        Single-row feature vector.
    top_n : int
        Number of top drivers to return per direction.

    Returns
    -------
    (positives, risks) : tuple[list[str], list[str]]
        Human-readable driver descriptions.
    """
    row_df = pd.DataFrame([X_row])
    shap_df = model.explain(row_df)  # type: ignore[union-attr]
    shap_row = shap_df.iloc[0].sort_values()

    positives: list[str] = []
    for feat in shap_row.nlargest(top_n).index:
        display = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())
        positives.append(f"{display} (+{shap_row[feat]:.3f})")

    risks: list[str] = []
    for feat in shap_row.nsmallest(top_n).index:
        display = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())
        risks.append(f"{display} ({shap_row[feat]:.3f})")

    return positives, risks
