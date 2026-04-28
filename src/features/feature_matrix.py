"""Boilerplate for the neighborhood feature matrix workstream."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features.zone_crosswalk import ZONE_TO_NTA, aggregate_nta_to_zone

logger = logging.getLogger(__name__)

_GEMINI_CACHE = Path("data/processed/gemini_labels_full.csv")
_GEMINI_FALLBACK_TIME_KEY = 2024


@dataclass(frozen=True)
class FeatureTable:
    """Metadata for feature tables owned by separate team members."""

    name: str
    owner: str
    join_keys: tuple[str, ...]


def normalize_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a feature matrix by filling NaN and clipping outliers.

    - Fills NaN with 0.0 for numeric columns.
    - Clips numeric columns to [-3, 3] standard deviations from mean (robust).

    Parameters
    ----------
    df:
        Feature DataFrame to normalize.

    Returns
    -------
    Normalized DataFrame (copy).
    """
    result = df.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0.0)
    for col in numeric_cols:
        mean = result[col].mean()
        std = result[col].std()
        if std and not pd.isna(std):
            result[col] = result[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)
    return result


def build_feature_matrix(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join a dictionary of source feature tables on shared keys."""

    tables = [frame for frame in feature_tables.values() if not frame.empty]
    if not tables:
        return pd.DataFrame(columns=["zone_id", "time_key"])

    merged = tables[0].copy()
    for frame in tables[1:]:
        join_keys = [
            column
            for column in ("zone_id", "time_key")
            if column in merged.columns and column in frame.columns
        ]
        merged = merged.merge(frame, how="outer", on=join_keys)
    return merged


# ---------------------------------------------------------------------------
# Zone-year panel builder (Phase 1)
# ---------------------------------------------------------------------------


def build_zone_year_matrix(
    etl_outputs: dict[str, pd.DataFrame],
    crosswalk: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build a (zone_id, year) panel from raw ETL outputs.

    Calls existing feature builders on appropriate ETL datasets, aggregates
    NTA-level results to zone-level via the crosswalk, and joins all on
    ``(zone_id, time_key)``.

    Parameters
    ----------
    etl_outputs:
        Dict keyed by dataset name (e.g. ``"licenses"``, ``"pluto"``),
        values are DataFrames from ``run_etl()``.
    crosswalk:
        Optional override for zone-to-NTA mapping.  Defaults to
        :data:`src.features.zone_crosswalk.ZONE_TO_NTA`.

    Returns
    -------
    Merged panel DataFrame with one row per (zone_id, time_key).
    """
    from src.features.demand_signals import build_demand_features
    from src.features.license_velocity import build_license_velocity_features
    from src.features.rent_trajectory import build_rent_trajectory_features

    if crosswalk is None:
        crosswalk = ZONE_TO_NTA

    feature_tables: dict[str, pd.DataFrame] = {}
    inspections_df = etl_outputs.get("inspections", pd.DataFrame())

    def _agg_to_zone(
        nta_df: pd.DataFrame, agg_rules: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """Aggregate NTA-level feature table to micro-zone level via crosswalk.

        Feature builders output zone_id = NTA code. This converts them to
        micro-zone IDs (bk-tandon, mn-fidi, etc.) by mapping NTA → zone
        and aggregating.
        """
        if nta_df.empty or "zone_id" not in nta_df.columns:  # pragma: no cover
            return nta_df
        # Rename zone_id (which is actually NTA) to nta_id for the crosswalk
        renamed = nta_df.rename(columns={"zone_id": "nta_id"})
        return aggregate_nta_to_zone(renamed, zone_col="nta_id", agg_rules=agg_rules)

    # --- License velocity (needs "licenses" dataset) ---
    licenses_df = etl_outputs.get("licenses", pd.DataFrame())
    if not licenses_df.empty:
        lv = build_license_velocity_features(licenses_df)
        if not lv.empty:
            lv_zone = _agg_to_zone(
                lv,
                agg_rules={
                    "license_velocity": "sum",
                    "net_opens": "sum",
                    "net_closes": "sum",
                },
            )
            if not lv_zone.empty:
                feature_tables["license_velocity"] = lv_zone

    # --- Rent trajectory (needs "pluto" dataset) ---
    # PLUTO is cross-sectional so rent_trajectory has no time_key.
    # We aggregate to zone level and store separately for a cross-join later.
    pluto_df = etl_outputs.get("pluto", pd.DataFrame())
    rent_static: pd.DataFrame | None = None
    if not pluto_df.empty:
        rt = build_rent_trajectory_features(pluto_df)
        if not rt.empty:
            rt_zone = _agg_to_zone(
                rt,
                agg_rules={
                    "rent_pressure": "mean",
                    "mean_assessed_value": "mean",
                },
            )
            if not rt_zone.empty:
                rent_static = rt_zone

    # --- Demand signals (needs "yelp" + "complaints_311") ---
    yelp_df = etl_outputs.get("yelp", pd.DataFrame())
    complaints_311_df = etl_outputs.get("complaints_311", pd.DataFrame())
    review_locations = _build_restaurant_zone_lookup(inspections_df)
    review_frame = _prepare_review_signals(
        yelp_df, restaurant_locations=review_locations
    )
    social_frame = _prepare_social_signals(complaints_311_df)
    if not review_frame.empty or not social_frame.empty:
        ds = build_demand_features(review_frame, social_frame)
        if not ds.empty:
            ds_zone = _agg_to_zone(ds)
            if not ds_zone.empty:
                feature_tables["demand_signals"] = ds_zone

    # --- ACS demographics (aggregate NTA -> zone) ---
    acs_df = etl_outputs.get("acs", pd.DataFrame())
    if not acs_df.empty:
        acs_zone = aggregate_nta_to_zone(
            acs_df,
            zone_col="nta_id",
            agg_rules={
                "population": "sum",
                "median_income": "mean",
                "rent_burden": "mean",
            },
        )
        if not acs_zone.empty:
            feature_tables["acs"] = acs_zone

    # --- Inspections: grade distribution per zone ---
    if not inspections_df.empty and "grade" in inspections_df.columns:
        insp = inspections_df.copy()
        insp["inspection_date"] = pd.to_datetime(
            insp["inspection_date"], errors="coerce"
        )
        insp = insp.dropna(subset=["inspection_date", "nta_id"])
        insp["time_key"] = insp["inspection_date"].dt.year.astype(int)
        insp["is_a"] = (insp["grade"] == "A").astype(int)
        grade_agg = insp.groupby(["nta_id", "time_key"], as_index=False).agg(
            inspection_grade_avg=("is_a", "mean"),
            restaurant_count=("restaurant_id", "nunique"),
        )
        grade_zone = aggregate_nta_to_zone(
            grade_agg,
            zone_col="nta_id",
            agg_rules={"inspection_grade_avg": "mean", "restaurant_count": "sum"},
        )
        if not grade_zone.empty:
            feature_tables["inspections"] = grade_zone

    # --- Permits: construction velocity (needs "permits" dataset) ---
    permits_df = etl_outputs.get("permits", pd.DataFrame())
    if (
        not permits_df.empty
        and "nta_id" in permits_df.columns
        and "job_count" in permits_df.columns
    ):
        p = permits_df.copy()
        p["permit_date"] = pd.to_datetime(p["permit_date"], errors="coerce")
        p = p.dropna(subset=["permit_date"])
        p["time_key"] = p["permit_date"].dt.year.astype(int)
        pv = p.groupby(["nta_id", "time_key"], as_index=False).agg(
            permit_velocity=("job_count", "sum")
        )
        pv_zone = aggregate_nta_to_zone(
            pv, zone_col="nta_id", agg_rules={"permit_velocity": "sum"}
        )
        if not pv_zone.empty:
            feature_tables["permits"] = pv_zone

    # --- Citi Bike: mobility proxy (needs "citibike" dataset) ---
    citibike_df = etl_outputs.get("citibike", pd.DataFrame())
    if not citibike_df.empty and "nta_id" in citibike_df.columns:
        cb = citibike_df.copy()
        if "year" in cb.columns and "time_key" not in cb.columns:
            cb = cb.rename(columns={"year": "time_key"})
        cb["time_key"] = (
            pd.to_numeric(cb["time_key"], errors="coerce").fillna(0).astype(int)
        )
        cb_zone = aggregate_nta_to_zone(
            cb,
            zone_col="nta_id",
            agg_rules={"trip_count": "sum", "station_count": "sum"},
        )
        if not cb_zone.empty:
            feature_tables["citibike"] = cb_zone

    # --- Airbnb: housing pressure static covariate ---
    airbnb_df = etl_outputs.get("airbnb", pd.DataFrame())
    airbnb_static: pd.DataFrame | None = None
    if not airbnb_df.empty and "nta_id" in airbnb_df.columns:
        ab = airbnb_df.copy()
        ab_zone = aggregate_nta_to_zone(
            ab,
            zone_col="nta_id",
            agg_rules={"listing_count": "sum", "entire_home_ratio": "mean"},
        )
        if not ab_zone.empty:
            airbnb_static = ab_zone

    if not feature_tables and rent_static is None and airbnb_static is None:
        return pd.DataFrame(columns=["zone_id", "time_key"])

    if feature_tables:
        merged = build_feature_matrix(feature_tables)
    else:
        merged = pd.DataFrame(columns=["zone_id", "time_key"])

    # Cross-join static rent features onto every (zone_id, time_key) row
    if rent_static is not None and not merged.empty:
        merged = merged.merge(rent_static, on="zone_id", how="left")
    elif rent_static is not None:
        merged = rent_static  # only static features available

    if airbnb_static is not None and not merged.empty:
        merged = merged.merge(airbnb_static, on="zone_id", how="left")
    elif airbnb_static is not None:
        merged = merged.merge(
            airbnb_static, on="zone_id", how="outer"
        )  # pragma: no cover

    # --- Upgrade healthy_review_share with Gemini labels if cache exists ---
    gemini_features = _load_gemini_review_features(yelp_df, review_locations)
    if not gemini_features.empty and "zone_id" in gemini_features.columns:
        gemini_ids = set(gemini_features["zone_id"].dropna().astype(str))
        gemini_zone = (
            gemini_features
            if gemini_ids.intersection(crosswalk.keys())
            else _agg_to_zone(gemini_features)
        )
        if not gemini_zone.empty:
            join_keys = [
                column
                for column in ("zone_id", "time_key")
                if column in merged.columns and column in gemini_zone.columns
            ]
            if merged.empty or not join_keys:
                merged = gemini_zone
            else:
                overlap_cols = [
                    column
                    for column in gemini_zone.columns
                    if column not in join_keys and column in merged.columns
                ]
                if overlap_cols:
                    merged = merged.drop(columns=overlap_cols)
                merged = merged.merge(gemini_zone, on=join_keys, how="outer")

    return merged


def _build_restaurant_zone_lookup(location_df: pd.DataFrame) -> pd.DataFrame:
    """Build a best-effort restaurant_id -> NTA lookup for review enrichment."""
    if (
        location_df.empty
        or "restaurant_id" not in location_df.columns
        or (
            "nta_id" not in location_df.columns and "zone_id" not in location_df.columns
        )
    ):
        return pd.DataFrame(columns=["restaurant_id", "zone_id"])

    zone_col = "nta_id" if "nta_id" in location_df.columns else "zone_id"
    subset = location_df.copy()
    if "inspection_date" in subset.columns:
        subset["inspection_date"] = pd.to_datetime(
            subset["inspection_date"], errors="coerce"
        )
        subset = subset.sort_values("inspection_date", ascending=False)
    subset["restaurant_id"] = (
        subset["restaurant_id"].replace({"UNKNOWN": pd.NA, "": pd.NA}).astype("string")
    )
    subset[zone_col] = (
        subset[zone_col].replace({"UNKNOWN": pd.NA, "": pd.NA}).astype("string")
    )
    subset = subset.dropna(subset=["restaurant_id", zone_col])
    subset = subset.rename(columns={zone_col: "zone_id"})
    return (
        subset[["restaurant_id", "zone_id"]]
        .drop_duplicates(subset=["restaurant_id"])
        .reset_index(drop=True)
    )


def _prepare_review_signals(
    yelp_df: pd.DataFrame,
    restaurant_locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert raw Yelp reviews into zone_id/time_key review signals."""
    if yelp_df.empty or "review_text" not in yelp_df.columns:
        return pd.DataFrame(columns=["zone_id", "time_key", "healthy_review_share"])

    df = yelp_df.copy()
    df["review_date"] = pd.to_datetime(df.get("review_date"), errors="coerce")
    df = df.dropna(subset=["review_date"])
    df["time_key"] = df["review_date"].dt.year

    if (
        "zone_id" not in df.columns
        and "nta_id" not in df.columns
        and restaurant_locations is not None
        and not restaurant_locations.empty
        and "restaurant_id" in df.columns
    ):
        df["restaurant_id"] = (
            df["restaurant_id"].replace({"UNKNOWN": pd.NA, "": pd.NA}).astype("string")
        )
        df = df.merge(restaurant_locations, on="restaurant_id", how="left")

    # If there's no zone_id or nta_id, we can't group spatially
    if "zone_id" not in df.columns and "nta_id" not in df.columns:
        return pd.DataFrame(columns=["zone_id", "time_key", "healthy_review_share"])

    id_col = "zone_id" if "zone_id" in df.columns else "nta_id"
    _healthy_kw = (
        r"(?<!\bun)\bhealthy\b|\bvegan\b|\borganic\b|\bsalad\b"
        r"|\bgrain[\s_-]?bowl\b|\bsmoothie\b|\bgluten[\s_-]?free\b"
        r"|\bvegetarian\b|\bnutritious\b|\bplant[\s_-]?based\b"
    )
    df["_healthy"] = (
        df["review_text"]
        .fillna("")
        .str.lower()
        .str.contains(_healthy_kw, regex=True, na=False)
        .astype(int)
    )

    grouped = df.groupby([id_col, "time_key"], as_index=False).agg(
        total=("_healthy", "count"),
        healthy_count=("_healthy", "sum"),
    )
    grouped["healthy_review_share"] = (
        grouped["healthy_count"] / grouped["total"].clip(lower=1)
    ).clip(0, 1)
    grouped = grouped.rename(columns={id_col: "zone_id"})
    return grouped[["zone_id", "time_key", "healthy_review_share"]]


_CD_TO_ZONE: dict[str, str | None] = {
    "Brooklyn": "BK09",
    "Manhattan": "MN17",
    "Queens": "QN70",
    "Bronx": "BX44",
    "Harlem": "MN25",
    "Astoria": "QN35",
    "Flushing": "QN49",
    "Williamsburg": "BK73",
    "Bushwick": "BK21",
    "Flatbush": "BK38",
    "Greenpoint": "BK29",
    "Sunset Park": "BK54",
    "Jackson Heights": "QN27",
    "Bay Ridge": "BK43",
    "Ridgewood": "QN17",
    "Unknown": None,
}


def _prepare_social_signals(
    complaints_311_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert 311 complaint data into zone_id/time_key social buzz signals."""
    _empty = pd.DataFrame(columns=["zone_id", "time_key", "social_buzz"])

    if (
        complaints_311_df is not None
        and not complaints_311_df.empty
        and "community_district" in complaints_311_df.columns
    ):
        df = complaints_311_df.copy()
        if "year" in df.columns and "time_key" not in df.columns:
            df = df.rename(columns={"year": "time_key"})
        elif "time_key" not in df.columns and "month" in df.columns:
            df["time_key"] = pd.to_datetime(df["month"], errors="coerce").dt.year
        if "time_key" not in df.columns:
            return _empty
        df["time_key"] = pd.to_numeric(df["time_key"], errors="coerce")
        df = df.dropna(subset=["time_key"])
        df["time_key"] = df["time_key"].astype(int)
        count_col = "count" if "count" in df.columns else None
        if count_col:
            agg = df.groupby(["community_district", "time_key"], as_index=False)[
                count_col
            ].sum()
            agg = agg.rename(columns={count_col: "complaint_count"})
        else:
            agg = df.groupby(["community_district", "time_key"], as_index=False).size()
            agg = agg.rename(columns={"size": "complaint_count"})
        agg["social_buzz"] = (agg["complaint_count"] / 20.0).clip(upper=1.0)
        agg["zone_id"] = agg["community_district"].map(_CD_TO_ZONE.get)
        agg = agg.dropna(subset=["zone_id"])
        if agg.empty:
            return _empty
        return agg[["zone_id", "time_key", "social_buzz"]].reset_index(drop=True)

    return _empty


def _load_gemini_review_features(
    yelp_df: pd.DataFrame,
    restaurant_locations: pd.DataFrame,
) -> pd.DataFrame:
    if not _GEMINI_CACHE.exists():
        return pd.DataFrame()  # pragma: no cover
    try:
        from src.nlp.review_aggregates import aggregate_full_halal_review_features

        labels_df = pd.read_csv(_GEMINI_CACHE)
        if "zone_id" not in labels_df.columns and not restaurant_locations.empty:
            labels_df = labels_df.merge(  # pragma: no cover
                restaurant_locations,
                on="restaurant_id",
                how="left",
            )
        if "time_key" not in labels_df.columns:
            labels_df["time_key"] = _GEMINI_FALLBACK_TIME_KEY  # pragma: no cover
        return aggregate_full_halal_review_features(labels_df)
    except Exception:
        logger.warning(
            "feature_matrix: failed to load Gemini review features", exc_info=True
        )
        return pd.DataFrame()
