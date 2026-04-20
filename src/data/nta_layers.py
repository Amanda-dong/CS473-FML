"""Load NTA boundary layers for spatial joins (Yelp → NTA → micro-zone_id)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

RAW_DIR = Path("data/raw")
NYC_2010_PATH = RAW_DIR / "nta_nyc_2010.geojson"


def load_nyc_ntas_for_zones() -> gpd.GeoDataFrame:
    """Load **2010** NYC NTA polygons (all boroughs) with ACS ``nta`` codes (MN22, BK09, …).

    Use this for Yelp → micro-zone assignment when the ACS GDB used by
    ``build_nta_features._load_manhattan_ntas`` is not available.

    Requires ``nta_nyc_2010.geojson`` from ``scripts/download_nta_geojson.py``.
    """
    if not NYC_2010_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {NYC_2010_PATH}. Run: python scripts/download_nta_geojson.py"
        )
    gdf = gpd.read_file(NYC_2010_PATH)
    if "nta" not in gdf.columns:
        raise ValueError("Expected column 'nta' (NTACode) in nta_nyc_2010.geojson")
    gdf["nta"] = gdf["nta"].astype(str).str.strip().str.upper()
    return gdf
