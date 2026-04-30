"""Lightweight helpers for describing recommendation micro-zones."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NTA_GEOJSON_CANDIDATES = [
    "data/raw/nta.geojson",
    "data/raw/nta2020_nyc.geojson",
]


@lru_cache(maxsize=1)
def _load_nta_polygons() -> tuple[list[str], Any]:
    """Load 2020 NTA polygons from local GeoJSON into shapely geometries.

    Returns (nta_codes, STRtree) or ([], None) if shapely/geojson unavailable.
    """
    try:
        import shapely
        from shapely.geometry import shape
    except ImportError:
        logger.warning(
            "shapely not available; lat_lon_to_nta will return empty strings"
        )
        return [], None

    for candidate in _NTA_GEOJSON_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", candidate, exc)
            continue

        codes: list[str] = []
        geoms: list[Any] = []
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            code = str(props.get("nta2020", "")).strip()
            if not code:
                continue
            try:
                geoms.append(shape(feature["geometry"]))
                codes.append(code)
            except Exception:
                continue

        if codes:
            tree = shapely.STRtree(geoms)
            logger.info("Loaded %d NTA polygons from %s", len(codes), candidate)
            return codes, tree

    logger.warning("No NTA GeoJSON found; lat_lon_to_nta will return empty strings")
    return [], None


def describe_microzone(zone_type: str, label: str) -> str:
    """Return a human-readable description for the UI and docs."""
    descriptions = {
        "campus_walkshed": f"{label} 10-minute campus walkshed",
        "lunch_corridor": f"{label} lunch corridor",
        "transit_catchment": f"{label} transit catchment",
        "business_district": f"{label} business district",
    }
    return descriptions.get(zone_type, label)


def lat_lon_to_nta(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Map lat/lon coordinates to 2020 NYC NTA codes via point-in-polygon.

    Uses an STRtree spatial index for efficient batch lookup against the 2020
    NTA boundary GeoJSON. Points not contained in any polygon (e.g. water) fall
    back to the nearest NTA centroid. Returns empty strings when the GeoJSON is
    unavailable.
    """
    codes, tree = _load_nta_polygons()
    if not codes or tree is None:
        return pd.Series("", index=lat.index)

    import shapely

    lat_arr = lat.to_numpy(dtype=float)
    lon_arr = lon.to_numpy(dtype=float)

    # Build point geometries: shapely expects (x=lon, y=lat)
    pts = shapely.points(lon_arr, lat_arr)

    # Batch point-in-polygon: returns (point_indices, polygon_indices)
    pt_idx, poly_idx = tree.query(pts, predicate="within")

    result = np.full(len(lat_arr), "", dtype=object)
    result[pt_idx] = np.array(codes)[poly_idx]

    # Fallback for unmatched points: nearest NTA centroid
    unmatched_mask = result == ""
    if unmatched_mask.any():
        try:
            poly_geoms = tree.geometries
            centroids = shapely.centroid(poly_geoms)
            cx = shapely.get_x(centroids)
            cy = shapely.get_y(centroids)
            um_lon = lon_arr[unmatched_mask]
            um_lat = lat_arr[unmatched_mask]
            # Euclidean distance in degree space (good enough for NYC scale)
            dists = (cx[:, None] - um_lon[None, :]) ** 2 + (
                cy[:, None] - um_lat[None, :]
            ) ** 2
            nearest = np.argmin(dists, axis=0)
            result[unmatched_mask] = np.array(codes)[nearest]
        except Exception as exc:
            logger.warning("NTA centroid fallback failed: %s", exc)

    return pd.Series(result, index=lat.index)
