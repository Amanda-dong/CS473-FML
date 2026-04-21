"""Assign Yelp businesses to micro-zones (NYC NTA → zone_id)."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.features.zone_crosswalk import resolve_nta_to_zone_id


def spatial_join_to_nta(
    frame: pd.DataFrame,
    *,
    lat_col: str,
    lng_col: str,
    nta_gdf: gpd.GeoDataFrame,
    how: str = "left",
) -> gpd.GeoDataFrame:
    """Join point rows to NTA polygons; ``nta`` column comes from ``nta_gdf``."""
    usable = frame.copy()
    usable = usable.dropna(subset=[lat_col, lng_col])
    point_gdf = gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable[lng_col], usable[lat_col]),
        crs="EPSG:4326",
    )
    target = nta_gdf[["nta", "geometry"]].copy()
    joined = gpd.sjoin(point_gdf, target, how=how, predicate="within")
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    return joined


def assign_yelp_business_zones(
    yelp_business: pd.DataFrame,
    nta_gdf: gpd.GeoDataFrame,
    *,
    id_col: str = "id",
    lat_col: str = "latitude",
    lng_col: str = "longitude",
) -> pd.DataFrame:
    """Add ``nta``, ``zone_id``, and flags for each Yelp business row.

    Parameters
    ----------
    yelp_business:
        Raw Yelp business export (must include ``id``, lat/lon).
    nta_gdf:
        NYC NTA polygons with ``nta`` column (ACS-style codes), e.g.
        :func:`src.data.nta_layers.load_nyc_ntas_for_zones`.

    Returns
    -------
    DataFrame with columns including ``restaurant_id``, ``nta``, ``zone_id``,
    ``in_nyc_nta``, ``in_modeled_microzone``.
    """
    if id_col not in yelp_business.columns:
        raise ValueError(f"Missing column {id_col!r}")

    base = yelp_business.copy()
    base["restaurant_id"] = base[id_col].astype(str).str.strip()

    joined = spatial_join_to_nta(
        base,
        lat_col=lat_col,
        lng_col=lng_col,
        nta_gdf=nta_gdf,
        how="left",
    )

    if isinstance(joined, gpd.GeoDataFrame):
        out = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))
    else:
        out = joined

    out["in_nyc_nta"] = out["nta"].notna() & (out["nta"].astype(str).str.strip() != "")
    out["zone_id"] = out["nta"].map(
        lambda x: resolve_nta_to_zone_id(x) if pd.notna(x) else None
    )
    out["in_modeled_microzone"] = out["zone_id"].notna()

    keep = [
        "restaurant_id",
        lat_col,
        lng_col,
        "nta",
        "zone_id",
        "in_nyc_nta",
        "in_modeled_microzone",
    ]
    present = [c for c in keep if c in out.columns]
    result = out[present].reset_index(drop=True)
    if "restaurant_id" in result.columns and result["restaurant_id"].duplicated().any():
        result = result.drop_duplicates(
            subset=["restaurant_id"], keep="first"
        ).reset_index(drop=True)
    return result
