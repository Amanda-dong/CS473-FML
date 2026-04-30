"""Load NTA boundary layers for spatial joins (Yelp → NTA → micro-zone_id)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

RAW_DIR = Path("data/raw")
NYC_2020_PATH = RAW_DIR / "nta.geojson"


def load_nyc_ntas_for_zones() -> gpd.GeoDataFrame:
    """Load **2020** NYC NTA polygons (all boroughs) with ACS ``nta2020`` codes (MN0202, BK0202, …).

    Use this for Yelp → micro-zone assignment when the ACS GDB used by
    ``build_nta_features._load_manhattan_ntas`` is not available.

    Requires ``nta.geojson`` from ``scripts/download_nta_geojson.py``.
    """
    if not NYC_2020_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {NYC_2020_PATH}. Run: python scripts/download_nta_geojson.py"
        )
    gdf = gpd.read_file(NYC_2020_PATH)
    if "nta2020" not in gdf.columns:
        raise ValueError("Expected column 'nta2020' in nta.geojson")
    gdf["nta2020"] = gdf["nta2020"].astype(str).str.strip().str.upper()
    return gdf
