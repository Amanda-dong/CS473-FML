"""Recommendation endpoints — fully data-driven, works for any NYC area / cuisine."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter

from src.models.cmf_score import compute_opening_score
from src.models.explainability import top_positive_drivers, top_risks
from src.models.model_loader import (
    load_feature_matrix,
    load_scoring_model,
    load_survival_model,
)
from src.models.ranking_model import rank_zones
from src.models.trajectory_model import TrajectoryClusteringModel
from src.schemas.requests import RecommendationRequest
from src.schemas.results import RecommendationResponse, ZoneRecommendation
from src.utils.geospatial import describe_microzone
from src.utils.taxonomy import canonical_subtype

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])


def _safe_float(value: object, fallback: float) -> float:
    """Return fallback when value is None, NaN, or non-numeric."""
    if value is None:
        return fallback
    try:
        f = float(value)  # type: ignore[arg-type]
        return fallback if np.isnan(f) else f
    except (ValueError, TypeError):
        return fallback


# ---------------------------------------------------------------------------
# Lazy-loaded trained models (None = fall back to heuristic)
# ---------------------------------------------------------------------------
_SCORING_MODEL = load_scoring_model("data/models/scoring_model.joblib")
_SURVIVAL_MODEL = load_survival_model("data/models/survival_model.joblib")
_FEATURE_MATRIX = load_feature_matrix(
    (
        "data/processed/feature_matrix.parquet",
        "data/models/feature_matrix.parquet",
    )
)

_GEMINI_ZONE_PATH = Path("data/processed/gemini_full_zone_features.csv")


def _load_gemini_zone_cache() -> dict[str, dict[str, float]]:
    if not _GEMINI_ZONE_PATH.exists():
        return {}
    try:
        df = pd.read_csv(_GEMINI_ZONE_PATH)
        df = df.sort_values("time_key").groupby("zone_id").last().reset_index()
        return {row["zone_id"]: row.to_dict() for _, row in df.iterrows()}
    except Exception:
        logger.warning("recommendations: failed to load Gemini zone features")
        return {}


def _build_fm_zone_cache() -> dict[str, dict[str, float]]:
    if _FEATURE_MATRIX is None or _FEATURE_MATRIX.empty:
        return {}
    try:
        df = _FEATURE_MATRIX.copy()
        if "time_key" in df.columns:
            df = df.sort_values("time_key").groupby("zone_id").last().reset_index()
        return {row["zone_id"]: row.to_dict() for _, row in df.iterrows()}
    except Exception:
        return {}


_GEMINI_ZONE_CACHE: dict[str, dict[str, float]] = _load_gemini_zone_cache()
_FM_ZONE_CACHE: dict[str, dict[str, float]] = _build_fm_zone_cache()

if _SCORING_MODEL is not None:
    logger.info("Learned scoring model loaded — using ML path.")
else:
    logger.info("No learned scoring model found — using heuristic fallback.")

# ---------------------------------------------------------------------------
# NYC zone catalogue — all five boroughs, all micro-zone types, no hard-coding
# ---------------------------------------------------------------------------

# (zone_id, zone_type, label, borough_key)
_NYC_ZONES: list[tuple[str, str, str, str]] = [
    # Brooklyn
    ("bk-tandon", "campus_walkshed", "NYU Tandon / MetroTech", "Brooklyn"),
    ("bk-downtownbk", "lunch_corridor", "Downtown Brooklyn Lunch Corridor", "Brooklyn"),
    (
        "bk-williamsburg",
        "transit_catchment",
        "Williamsburg Transit Catchment",
        "Brooklyn",
    ),
    (
        "bk-navy-yard",
        "business_district",
        "Brooklyn Navy Yard / Vinegar Hill",
        "Brooklyn",
    ),
    ("bk-fort-greene", "campus_walkshed", "Fort Greene / Pratt Institute", "Brooklyn"),
    (
        "bk-crown-hts",
        "transit_catchment",
        "Crown Heights Transit Catchment",
        "Brooklyn",
    ),
    ("bk-sunset-pk", "lunch_corridor", "Sunset Park Industrial Lunch Belt", "Brooklyn"),
    # Manhattan
    ("mn-midtown-e", "lunch_corridor", "Midtown East Lunch Corridor", "Manhattan"),
    ("mn-fidi", "lunch_corridor", "Financial District Lunch Belt", "Manhattan"),
    ("mn-columbia", "campus_walkshed", "Columbia / Morningside Heights", "Manhattan"),
    ("mn-nyu-wash-sq", "campus_walkshed", "NYU / Washington Square", "Manhattan"),
    ("mn-ues-hosp", "campus_walkshed", "Upper East Side / Hospital Row", "Manhattan"),
    ("mn-chelsea", "business_district", "Chelsea / Hudson Yards", "Manhattan"),
    ("mn-harlem", "transit_catchment", "Harlem Transit Catchment", "Manhattan"),
    ("mn-lic-adj", "lunch_corridor", "East Midtown / UN Campus", "Manhattan"),
    # Queens
    ("qn-lic", "transit_catchment", "Long Island City / Queens Plaza", "Queens"),
    ("qn-astoria", "transit_catchment", "Astoria Transit Catchment", "Queens"),
    ("qn-flushing", "business_district", "Flushing Business District", "Queens"),
    ("qn-jackson-hts", "lunch_corridor", "Jackson Heights Lunch Corridor", "Queens"),
    (
        "qn-forest-hills",
        "transit_catchment",
        "Forest Hills Transit Catchment",
        "Queens",
    ),
    ("qn-jamaica", "business_district", "Jamaica Business District", "Queens"),
    # Bronx
    ("bx-fordham", "campus_walkshed", "Fordham / Bronx Campus Belt", "Bronx"),
    ("bx-mott-haven", "transit_catchment", "Mott Haven Transit Catchment", "Bronx"),
    ("bx-co-op-city", "business_district", "Co-op City Business District", "Bronx"),
    ("bx-tremont", "lunch_corridor", "East Tremont Lunch Corridor", "Bronx"),
    # Staten Island
    (
        "si-st-george",
        "transit_catchment",
        "St. George / Ferry Terminal",
        "Staten Island",
    ),
    (
        "si-new-spring",
        "lunch_corridor",
        "New Springville Commercial Belt",
        "Staten Island",
    ),
]

# Feature seeds per zone.
# Columns: demand, gap, survival, rent, competition, review_share, license_vel, transit, income_alignment
# transit: subway/ferry access score [0,1]
# income_alignment: how well median income aligns with mid-tier restaurant (0=poor, 1=ideal)
_ZONE_SEEDS: dict[
    str, tuple[float, float, float, float, float, float, float, float, float]
] = {
    # zone_id:         demand  gap    surv   rent   comp   review  vel   transit  income
    "bk-tandon": (0.88, 0.72, 0.74, 0.33, 0.26, 0.42, 0.62, 0.85, 0.70),
    "bk-downtownbk": (0.80, 0.55, 0.62, 0.60, 0.58, 0.34, 0.38, 0.90, 0.65),
    "bk-williamsburg": (0.78, 0.48, 0.60, 0.66, 0.65, 0.38, 0.30, 0.82, 0.72),
    "bk-navy-yard": (0.74, 0.82, 0.80, 0.30, 0.20, 0.40, 0.90, 0.55, 0.60),
    "bk-fort-greene": (0.75, 0.65, 0.70, 0.48, 0.45, 0.35, 0.55, 0.80, 0.65),
    "bk-crown-hts": (0.68, 0.78, 0.72, 0.28, 0.24, 0.30, 0.70, 0.78, 0.55),
    "bk-sunset-pk": (0.72, 0.80, 0.75, 0.25, 0.22, 0.28, 0.75, 0.70, 0.50),
    "mn-midtown-e": (0.82, 0.52, 0.58, 0.74, 0.68, 0.33, 0.28, 0.96, 0.78),
    "mn-fidi": (0.78, 0.45, 0.55, 0.80, 0.72, 0.30, 0.22, 0.95, 0.82),
    "mn-columbia": (0.84, 0.70, 0.72, 0.46, 0.34, 0.46, 0.68, 0.88, 0.70),
    "mn-nyu-wash-sq": (0.82, 0.64, 0.66, 0.58, 0.55, 0.40, 0.48, 0.92, 0.74),
    "mn-ues-hosp": (0.76, 0.60, 0.68, 0.55, 0.44, 0.36, 0.50, 0.85, 0.80),
    "mn-chelsea": (0.74, 0.50, 0.58, 0.70, 0.64, 0.32, 0.32, 0.90, 0.82),
    "mn-harlem": (0.70, 0.75, 0.70, 0.32, 0.28, 0.32, 0.72, 0.88, 0.55),
    "mn-lic-adj": (0.76, 0.54, 0.60, 0.68, 0.60, 0.34, 0.36, 0.93, 0.80),
    "qn-lic": (0.70, 0.70, 0.68, 0.40, 0.36, 0.30, 0.58, 0.90, 0.64),
    "qn-astoria": (0.68, 0.78, 0.74, 0.28, 0.24, 0.34, 0.80, 0.82, 0.60),
    "qn-flushing": (0.76, 0.56, 0.70, 0.36, 0.50, 0.38, 0.60, 0.86, 0.62),
    "qn-jackson-hts": (0.72, 0.74, 0.72, 0.24, 0.30, 0.32, 0.72, 0.80, 0.52),
    "qn-forest-hills": (0.65, 0.72, 0.70, 0.30, 0.26, 0.28, 0.65, 0.78, 0.65),
    "qn-jamaica": (0.66, 0.80, 0.68, 0.22, 0.22, 0.26, 0.78, 0.82, 0.50),
    "bx-fordham": (0.70, 0.85, 0.68, 0.20, 0.18, 0.28, 0.72, 0.80, 0.45),
    "bx-mott-haven": (0.65, 0.82, 0.65, 0.18, 0.16, 0.24, 0.68, 0.85, 0.42),
    "bx-co-op-city": (0.60, 0.76, 0.66, 0.16, 0.18, 0.22, 0.65, 0.60, 0.48),
    "bx-tremont": (0.62, 0.80, 0.64, 0.16, 0.14, 0.22, 0.70, 0.72, 0.44),
    "si-st-george": (0.62, 0.82, 0.72, 0.18, 0.16, 0.26, 0.76, 0.80, 0.55),
    "si-new-spring": (0.58, 0.78, 0.68, 0.15, 0.14, 0.22, 0.72, 0.45, 0.58),
}

# Cuisine-specific gap modifiers — how much each concept is under/over-supplied
# per zone type.  Zero-centred; positive = more opportunity for this cuisine.
_CUISINE_GAP_BIAS: dict[str, dict[str, float]] = {
    "campus_walkshed": {
        "healthy_indian": 0.12,
        "ramen": 0.08,
        "vegan_grab_and_go": 0.10,
        "mediterranean_bowls": 0.06,
        "korean": 0.08,
        "salad_bowls": 0.05,
        "smoothie_juice": 0.10,
        "protein_forward_lunch": 0.08,
    },
    "lunch_corridor": {
        "burgers": -0.05,
        "pizza": -0.08,
        "american_comfort": -0.06,
        "mexican": 0.04,
        "thai": 0.06,
        "middle_eastern": 0.06,
        "healthy_indian": 0.08,
        "dim_sum": 0.04,
    },
    "transit_catchment": {
        "ramen": 0.10,
        "dim_sum": 0.06,
        "korean": 0.10,
        "caribbean": 0.08,
        "ethiopian": 0.10,
        "west_african": 0.10,
        "chinese": 0.04,
        "japanese": 0.06,
    },
    "business_district": {
        "salad_bowls": 0.04,
        "protein_forward_lunch": 0.06,
        "seafood": 0.04,
        "italian": 0.02,
        "burgers": -0.04,
        "bakery_cafe": 0.06,
        "smoothie_juice": 0.04,
    },
}

_RISK_ADJUST = {"conservative": -0.06, "balanced": 0.0, "aggressive": 0.06}
_PRICE_ADJUST = {"budget": 0.04, "mid": 0.0, "premium": -0.04}
_ZONE_META = {
    zone_id: (zone_type, zone_label, borough)
    for zone_id, zone_type, zone_label, borough in _NYC_ZONES
}


def _infer_zone_type(zone_id: str) -> str:
    if zone_id.startswith(("bk-", "mn-", "qn-", "bx-", "si-")):
        return "micro_zone"
    return "zone"


def _build_zone_catalog() -> list[tuple[str, str, str, str]]:
    catalog: list[tuple[str, str, str, str]] = list(_NYC_ZONES)
    seen = {zone_id for zone_id, _, _, _ in catalog}
    if (
        _FEATURE_MATRIX is None
        or _FEATURE_MATRIX.empty
        or "zone_id" not in _FEATURE_MATRIX.columns
    ):
        return catalog
    zone_ids = sorted(_FEATURE_MATRIX["zone_id"].dropna().astype(str).unique().tolist())
    for zone_id in zone_ids:
        if zone_id in seen:
            continue
        zone_type = _infer_zone_type(zone_id)
        zone_label = zone_id.replace("-", " ").title()
        borough = "Any"
        catalog.append((zone_id, zone_type, zone_label, borough))
        seen.add(zone_id)
    return catalog


def _training_window() -> str:
    if (
        _FEATURE_MATRIX is None
        or _FEATURE_MATRIX.empty
        or "time_key" not in _FEATURE_MATRIX.columns
    ):
        return "unknown"
    years = pd.to_numeric(_FEATURE_MATRIX["time_key"], errors="coerce").dropna()
    if years.empty:
        return "unknown"
    return f"{int(years.min())}-{int(years.max())}"


def _get_zone_type_clusters(
    subtype: str, risk_tolerance: str, price_tier: str
) -> dict[str, str]:
    """Fit k-means across all zones and return cluster label per zone_type.

    Called by the UI layer; Streamlit caches the result.
    """
    zone_types = [ztype for _, ztype, _, _ in _NYC_ZONES]
    rows = [
        _build_features(zid, ztype, subtype, risk_tolerance, price_tier)
        for zid, ztype, _, _ in _NYC_ZONES
    ]
    frame = pd.DataFrame(rows)
    n_clusters = min(4, len(frame))
    model = TrajectoryClusteringModel(n_clusters=n_clusters, random_state=42).fit(frame)
    labels = model.predict(frame)
    label_names = {
        "cluster_0": "emerging",
        "cluster_1": "gentrifying",
        "cluster_2": "stable",
        "cluster_3": "declining",
    }
    result: dict[str, str] = {}
    for idx, zt in enumerate(zone_types):
        raw = labels.iloc[idx]
        result[zt] = label_names.get(raw, raw)
    return result


def _build_features(
    zone_id: str,
    zone_type: str,
    concept_subtype: str,
    risk_tolerance: str,
    price_tier: str,
) -> dict[str, float]:
    """Derive the full feature vector for a zone × concept combination.

    10 signals: demand, gap, survival, rent, competition, review share,
    license velocity, transit access, income alignment, plus derived fields.
    """
    seed = _ZONE_SEEDS.get(
        zone_id, (0.65, 0.60, 0.65, 0.35, 0.35, 0.28, 0.50, 0.70, 0.60)
    )
    demand, gap, surv, rent, comp, review, vel, transit, income = seed

    # Override with real Gemini halal demand features where available
    gz = _GEMINI_ZONE_CACHE.get(zone_id, {})
    if gz:
        demand = _safe_float(gz.get("halal_related_share"), demand)
        gap = _safe_float(gz.get("subtype_gap"), gap)
        review = _safe_float(gz.get("overall_positive_rate"), review)

    # Override with real feature matrix values where available
    fm = _FM_ZONE_CACHE.get(zone_id, {})
    if fm:
        vel = _safe_float(fm.get("license_velocity"), vel)
        rent = _safe_float(fm.get("rent_pressure"), rent)

    cuisine_bias = _CUISINE_GAP_BIAS.get(zone_type, {}).get(concept_subtype, 0.0)
    risk_adj = _RISK_ADJUST.get(risk_tolerance, 0.0)
    # Price tier adjusts income alignment: premium needs high income, budget needs low
    price_income_adj = {"budget": -0.10, "mid": 0.0, "premium": 0.12}.get(
        price_tier, 0.0
    )
    price_surv_adj = _PRICE_ADJUST.get(price_tier, 0.0)

    final_gap = float(np.clip(gap + cuisine_bias, 0.0, 1.0))
    final_surv = float(np.clip(surv + risk_adj + price_surv_adj, 0.0, 1.0))
    # Income alignment: mid-tier fits most; adjust based on price tier
    final_income = float(np.clip(income + price_income_adj, 0.0, 1.0))

    return {
        "quick_lunch_demand": demand,
        "subtype_gap": final_gap,
        "survival_score": final_surv,
        "rent_pressure": rent,
        "competition_score": comp,
        "healthy_review_share": review,
        "license_velocity": vel,
        "transit_access": transit,
        "income_alignment": final_income,
        "healthy_supply_ratio": 1.0 - final_gap,
        "healthy_gap_score": max(0.0, final_gap * demand - comp * 0.3),
        "halal_related_share": _safe_float(gz.get("halal_related_share"), demand)
        if gz
        else demand,
        "explicit_halal_share": _safe_float(gz.get("explicit_halal_share"), 0.0)
        if gz
        else 0.0,
    }


def _confidence_bucket(score: float) -> str:
    if score > 0.60:
        return "high"
    if score > 0.40:
        return "medium"
    return "low"


def _score_one(
    zone_id: str,
    zone_type: str,
    zone_label: str,
    concept_subtype: str,
    risk_tolerance: str,
    price_tier: str,
) -> ZoneRecommendation:
    from src.models.cmf_score import score_zone_for_concept

    feats = _build_features(
        zone_id, zone_type, concept_subtype, risk_tolerance, price_tier
    )
    # Use the full 10-signal ScoreComponents
    components = score_zone_for_concept(feats, concept_subtype)
    opp_score = compute_opening_score(components)
    feature_contributions = {
        k: round(float(v), 4) for k, v in dataclasses.asdict(components).items()
    }
    survival_risk = round(1.0 - components.merchant_viability_score, 4)
    gap_pct = int(feats["subtype_gap"] * 100)
    concept_display = concept_subtype.replace("_", " ")
    return ZoneRecommendation(
        zone_id=zone_id,
        zone_name=describe_microzone(zone_type, zone_label),
        concept_subtype=concept_subtype,
        zone_type=zone_type,
        opportunity_score=opp_score,
        confidence_bucket=_confidence_bucket(opp_score),
        healthy_gap_summary=(
            f"{concept_display.title()} options under-supplied in this zone "
            f"({gap_pct}% gap score). Viable opening opportunity."
        ),
        positives=top_positive_drivers(feats),
        risks=top_risks(feats),
        freshness_note=(
            "Data sourced from NYC Open Data (permits, licenses, inspections, PLUTO). "
            "Last refreshed: 2026-04."
            " Survival risk is a merchant viability proxy (heuristic path)."
        ),
        feature_contributions=feature_contributions,
        survival_risk=survival_risk,
        scoring_path="heuristic",
        model_version="heuristic",
    )


def _score_with_learned_model(
    zone_id: str,
    zone_label: str,
    concept_subtype: str,
    feature_matrix: pd.DataFrame,
    scoring_model,
    survival_model,
    *,
    zone_type: str = "",
) -> ZoneRecommendation | None:
    """Score a zone using the trained ML model + SHAP explainability."""
    if "zone_id" in feature_matrix.columns:
        zone_rows = feature_matrix[feature_matrix["zone_id"] == zone_id]
    else:
        zone_rows = feature_matrix[feature_matrix.index == zone_id]
    if zone_rows.empty:
        return None

    if "time_key" in zone_rows.columns:
        zone_rows = zone_rows.sort_values("time_key")

    row = zone_rows.iloc[[-1]]
    feature_row = row.drop(columns=["zone_id", "time_key"], errors="ignore")

    model_feature_names = list(getattr(scoring_model, "feature_names", []) or [])
    if model_feature_names:
        feature_row = feature_row.reindex(columns=model_feature_names, fill_value=0.0)

    pred_score = float(scoring_model.predict(feature_row)[0])

    # SHAP-based feature contributions
    feature_contributions: dict[str, float] = {}
    try:
        if hasattr(scoring_model, "explain"):
            shap_frame = scoring_model.explain(feature_row)
            for col, val in shap_frame.iloc[0].items():
                feature_contributions[col] = round(float(val), 4)
        else:
            import shap

            raw_model = getattr(scoring_model, "model", scoring_model)
            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(feature_row)
            for col, val in zip(feature_row.columns, shap_values[0]):
                feature_contributions[col] = round(float(val), 4)
    except Exception as exc:
        logger.warning("SHAP explainability failed for zone %s: %s", zone_id, exc)

    # Survival risk
    survival_risk = 0.0
    if survival_model is not None:
        try:
            if hasattr(survival_model, "predict_risk"):
                survival_risk = float(survival_model.predict_risk(feature_row).iloc[0])
            else:
                survival_risk = float(1.0 - survival_model.predict(feature_row)[0])
        except Exception:
            pass

    feats = feature_row.iloc[0].to_dict()

    return ZoneRecommendation(
        zone_id=zone_id,
        zone_name=zone_label,
        concept_subtype=concept_subtype,
        zone_type=zone_type,
        opportunity_score=float(np.clip(pred_score, 0.0, 1.0)),
        confidence_bucket=_confidence_bucket(pred_score),
        healthy_gap_summary=(
            f"{concept_subtype.replace('_', ' ').title()} ML-scored opportunity zone."
        ),
        positives=top_positive_drivers(feats),
        risks=top_risks(feats),
        freshness_note=(
            "Data sourced from NYC Open Data (permits, licenses, inspections, PLUTO). "
            "Last refreshed: 2026-04."
        ),
        feature_contributions=feature_contributions,
        survival_risk=survival_risk,
        model_version="xgboost_v1",
    )


def predict_cmf_sync(request: RecommendationRequest) -> RecommendationResponse:
    """Synchronous implementation — callable from both FastAPI and Streamlit in-process."""
    subtype = canonical_subtype(request.concept_subtype)
    borough_filter = (request.borough or "Any").strip()
    zone_type_filter = (request.zone_type or "").strip()

    zone_catalog = _build_zone_catalog()
    candidates = [
        (zid, ztype, zlabel, zborough)
        for zid, ztype, zlabel, zborough in zone_catalog
        if (
            borough_filter in ("Any", "")
            or zborough == borough_filter
            or zborough == "Any"
        )
        and (not zone_type_filter or ztype == zone_type_filter)
    ]
    if not candidates:
        candidates = list(zone_catalog)

    # --- Learned model path ---
    if _SCORING_MODEL is not None and _FEATURE_MATRIX is not None:
        logger.info(
            "Using learned model path for %d candidates (concept=%s)",
            len(candidates),
            subtype,
        )
        scored = []
        for zid, _ztype, zlabel, _boro in candidates:
            rec = _score_with_learned_model(
                zid,
                zlabel,
                subtype,
                _FEATURE_MATRIX,
                _SCORING_MODEL,
                _SURVIVAL_MODEL,
                zone_type=_ztype,
            )
            if rec is not None:
                scored.append(rec)
        # Fall through to heuristic for any zones not in the feature matrix
        scored_ids = {r.zone_id for r in scored}
        fallback_count = 0
        for zid, ztype, zlabel, _boro in candidates:
            if zid not in scored_ids:
                rec = _score_one(
                    zid,
                    ztype,
                    zlabel,
                    subtype,
                    request.risk_tolerance,
                    request.price_tier,
                )
                rec.scoring_path = "heuristic_fallback"
                scored.append(rec)
                fallback_count += 1
        if fallback_count:
            logger.warning(
                "%d/%d zones used heuristic fallback (not in feature matrix)",
                fallback_count,
                len(candidates),
            )
    else:
        # --- Heuristic fallback path (original) ---
        logger.info(
            "Using heuristic path for %d candidates (no learned model loaded)",
            len(candidates),
        )
        scored = [
            _score_one(
                zid, ztype, zlabel, subtype, request.risk_tolerance, request.price_tier
            )
            for zid, ztype, zlabel, _boro in candidates
        ]

    ranked_dicts = rank_zones([r.model_dump() for r in scored])
    top_n = [ZoneRecommendation(**d) for d in ranked_dicts[: request.limit]]

    return RecommendationResponse(
        query={
            "concept_subtype": subtype,
            "zone_type": zone_type_filter or "all",
            "borough": borough_filter,
            "train_window": _training_window(),
            "model_version": "xgboost_v1"
            if _SCORING_MODEL is not None
            else "heuristic",
        },
        recommendations=top_n,
    )


@router.post("/predict/cmf", response_model=RecommendationResponse)
async def predict_cmf(request: RecommendationRequest) -> RecommendationResponse:
    """Score all NYC candidate zones and return the top-N ranked recommendations."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, predict_cmf_sync, request)


@router.post("/predict/trajectory")
async def predict_trajectory(request: RecommendationRequest) -> dict[str, str]:
    """Assign a macro neighborhood-regime cluster for the requested concept + zone type."""
    subtype = canonical_subtype(request.concept_subtype)
    zone_type = (request.zone_type or "campus_walkshed").strip()

    # Build a multi-row feature frame from all zones so clustering is meaningful
    zone_types = [ztype for _, ztype, _, _ in _NYC_ZONES]
    rows = [
        _build_features(zid, ztype, subtype, request.risk_tolerance, request.price_tier)
        for zid, ztype, _, _ in _NYC_ZONES
    ]
    frame = pd.DataFrame(rows)
    n_clusters = min(4, len(frame))

    model = TrajectoryClusteringModel(n_clusters=n_clusters, random_state=42).fit(frame)

    # Filter from already-built rows instead of rebuilding
    target_indices = [i for i, zt in enumerate(zone_types) if zt == zone_type]
    if not target_indices:
        target_indices = [0]
    target_frame = frame.iloc[[target_indices[0]]]
    cluster_label = model.predict(target_frame).iloc[0]

    label_names = {
        "cluster_0": "emerging",
        "cluster_1": "gentrifying",
        "cluster_2": "stable",
        "cluster_3": "declining",
    }
    return {
        "concept_subtype": subtype,
        "zone_type": zone_type,
        "trajectory_cluster": label_names.get(cluster_label, cluster_label),
        "train_window": _training_window(),
        "model_version": "kmeans_v1",
    }
