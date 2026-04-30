"""ETL for NTA boundary and crosswalk assets.

Attempts to load from local GeoJSON; downloads from NYC Open Data if missing.
Falls back to a static NTA code list with null geometry if download fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

_GEOJSON_PATH = Path("data/geojson/nta_boundaries.geojson")
_NYC_OPEN_DATA_URL = (
    "https://data.cityofnewyork.us/api/geospatial/9nt8-h7nd?method=export&type=GeoJSON"
)

_STATIC_NTA_CODES = [
    "BK0102",
    "BK0202",
    "BK0203",
    "BK0261",
    "BK0401",
    "BK0402",
    "BK0702",
    "BK0703",
    "BK0802",
    "BK1001",
    "BK1101",
    "BK1401",
    "BK1403",
    "BX0101",
    "BX0102",
    "BX0201",
    "BX0301",
    "BX0503",
    "BX0602",
    "BX0701",
    "BX0801",
    "BX0803",
    "BX0904",
    "BX1002",
    "BX1004",
    "MN0101",
    "MN0102",
    "MN0201",
    "MN0202",
    "MN0401",
    "MN0502",
    "MN0604",
    "MN0801",
    "MN0901",
    "MN1001",
    "MN1002",
    "QN0101",
    "QN0102",
    "QN0103",
    "QN0201",
    "QN0301",
    "QN0401",
    "QN0402",
    "QN0504",
    "QN0601",
    "QN0602",
    "QN0701",
    "QN0702",
    "QN0704",
    "QN0707",
    "QN1201",
    "SI0101",
    "SI0204",
]

DATASET_SPEC = DatasetSpec(
    name="boundaries",
    owner="integration",
    spatial_unit="geometry",
    time_grain="static",
    description="NTA, community district, and micro-zone boundary assets.",
    columns=("zone_id", "zone_type", "geometry_wkt"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


def _static_fallback() -> pd.DataFrame:
    """Return a minimal DataFrame with the 30 representative NTA codes and null geometry."""
    return pd.DataFrame(
        {
            "zone_id": _STATIC_NTA_CODES,
            "zone_type": "nta",
            "geometry_wkt": None,
        }
    )


def _load_geojson(path: Path) -> pd.DataFrame:
    """Load NTA boundaries from GeoJSON using geopandas."""
    import geopandas as gpd  # type: ignore[import]

    gdf = gpd.read_file(str(path))

    # NYC Open Data uses different NTA field names across vintages.
    nta_field = next(
        (
            f
            for f in ("NTACode", "nta2020", "ntacode", "NTACode2020")
            if f in gdf.columns
        ),
        None,
    )
    if nta_field is None:
        for col in gdf.columns:
            if (
                gdf[col].dtype == object
                and gdf[col].str.match(r"^[A-Z]{2}\d{2}$").any()
            ):
                nta_field = col
                break

    if nta_field is None:
        logger.warning(
            "etl_boundaries: could not find NTA code field in %s", gdf.columns.tolist()
        )
        return _static_fallback()

    result = pd.DataFrame(
        {
            "zone_id": gdf[nta_field].astype(str),
            "zone_type": "nta",
            "geometry_wkt": gdf.geometry.to_wkt(),
        }
    )
    return result.dropna(subset=["zone_id"]).reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Load NTA boundary GeoJSON. Downloads from NYC Open Data if not present locally."""
    _GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _GEOJSON_PATH.exists():
        try:
            return _load_geojson(_GEOJSON_PATH)
        except ImportError:
            logger.warning(
                "etl_boundaries: geopandas not available — using static fallback"
            )
            return _static_fallback()
        except Exception as exc:
            logger.warning(
                "etl_boundaries: failed to parse local GeoJSON (%s) — downloading fresh",
                exc,
            )

    try:
        import requests

        logger.info("etl_boundaries: downloading NTA GeoJSON from NYC Open Data")
        resp = requests.get(_NYC_OPEN_DATA_URL, timeout=30)
        resp.raise_for_status()
        _GEOJSON_PATH.write_bytes(resp.content)
        return _load_geojson(_GEOJSON_PATH)
    except ImportError:
        logger.warning(
            "etl_boundaries: geopandas not available — using static fallback"
        )
        return _static_fallback()
    except Exception as exc:
        logger.warning(
            "etl_boundaries: download failed (%s) — using static fallback", exc
        )
        return _static_fallback()
