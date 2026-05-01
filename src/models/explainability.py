"""Turn zone feature dicts into human-readable recommendation-card explanations."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def top_positive_drivers(zone_features: Mapping[str, float]) -> list[str]:
    """Return quantitative explanation strings for the strongest positive signals."""
    drivers: list[str] = []

    halal_share = zone_features.get("halal_related_share", 0.0)
    if halal_share > 0.6:
        drivers.append(
            f"High daytime foot-traffic / halal-demand index ({halal_share:.0%})"
        )

    subtype_gap = zone_features.get("subtype_gap", 0.0)
    if subtype_gap > 0.5:
        drivers.append(
            f"Strong cuisine gap ({subtype_gap:.0%}) — "
            "this concept is under-supplied here"
        )

    survival_score = zone_features.get("target", 0.0)
    if survival_score > 0.6:
        drivers.append(
            f"Survival model gives {survival_score:.0%} commercial viability"
        )

    license_velocity = zone_features.get("license_velocity", 0.0)
    if license_velocity > 0.3:
        drivers.append("Positive license velocity — active neighborhood growth signal")

    overall_pos = zone_features.get("overall_positive_rate", 0.0)
    if overall_pos > 0.3:
        drivers.append(
            f"NLP review signals show {overall_pos:.0%} demand for this category"
        )

    trip_raw = float(zone_features.get("trip_count", 0.0))
    trip_norm = min(trip_raw / 200_000.0, 1.0)
    if trip_norm > 0.75:
        drivers.append(
            f"Excellent transit accessibility ({trip_norm:.0%}) — "
            "maximises foot-traffic"
        )

    income_raw = float(zone_features.get("median_income_static", 0.0))
    income_alignment = (
        min(max((income_raw - 30_000.0) / 170_000.0, 0.0), 1.0)
        if income_raw > 1.0
        else income_raw
    )
    if income_alignment > 0.65:
        drivers.append("Income profile aligns well with the chosen price tier")

    return drivers or ["Explanation rules not configured yet"]


def top_risks(zone_features: Mapping[str, float]) -> list[str]:
    """Return quantitative risk strings for recommendation cards."""
    risks: list[str] = []

    rent_pressure = zone_features.get("rent_pressure", 0.0)
    if rent_pressure > 0.5:
        risks.append(
            f"High rent pressure ({rent_pressure:.0%}) — "
            "may compress margins significantly"
        )

    comp_score = zone_features.get("restaurant_count_static", 0.0)
    comp_norm = min(float(comp_score) / 50.0, 1.0)
    if comp_norm > 0.5:
        risks.append(
            f"Saturated market ({comp_norm:.0%} competitor density) — "
            "differentiation required"
        )

    survival_score = zone_features.get("target", 0.0)
    if survival_score < 0.4:
        risks.append("Below-average survival outlook — consider more established zone")

    income_raw = float(zone_features.get("median_income_static", 0.0))
    income_alignment = (
        min(max((income_raw - 30_000.0) / 170_000.0, 0.0), 1.0)
        if income_raw > 1.0
        else income_raw
    )
    if income_alignment < 0.35:
        risks.append(
            "Income/price-tier mismatch — local spending power may not "
            "support this concept"
        )

    trip_count = zone_features.get("trip_count", 0.0)
    trip_norm = min(float(trip_count) / 200_000.0, 1.0)
    if trip_norm < 0.45:
        risks.append(
            "Limited transit access — foot-traffic relies on local residents only"
        )

    return risks or ["Risk rules not configured yet"]


# ---------------------------------------------------------------------------
# SHAP-based explainability (Phase 4)
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "halal_related_share": "Daytime foot-traffic demand",
    "subtype_gap": "Cuisine white-space opportunity",
    "target": "Commercial viability",
    "rent_pressure": "Rent pressure",
    "restaurant_count_static": "Market competition",
    "license_velocity": "Neighborhood growth signal",
    "overall_positive_rate": "NLP demand confirmation",
    "trip_count": "Transit accessibility",
    "median_income_static": "Price-tier income fit",
    "inspection_grade_avg_static": "Health inspection quality",
}


def shap_drivers(
    model: object, X_row: pd.Series, top_n: int = 3
) -> tuple[list[str], list[str]]:
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
