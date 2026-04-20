"""Download NYC NTA boundary GeoJSON for local use.

Writes:

- ``data/raw/nta2020_nyc.geojson`` — 2020 NTAs (all boroughs), from ArcGIS
  ``NYC_Neighborhood_Tabulation_Areas_2020`` (fields normalized to lowercase).

- ``data/raw/nta_nyc_2010.geojson`` — **2010** NTAs **all five boroughs**,
  from ``NYC_2010_NTA``. Uses ``NTACode`` (e.g. MN22, BK09) which matches
  ``zone_crosswalk.ZONE_TO_NTA``.

The legacy ``build_nta_features.load_manhattan_ntas()`` still expects
``nta.geojson`` + the ACS GDB crosswalk. For **Yelp → zone_id** without the GDB,
use ``load_nyc_ntas_for_zones()`` in ``src.data.nta_layers`` (reads
``nta_nyc_2010.geojson``).

Usage::

    python scripts/download_nta_geojson.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"

URL_2020 = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/"
    "NYC_Neighborhood_Tabulation_Areas_2020/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson&resultRecordCount=500"
)
URL_2010_NYC = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/"
    "NYC_2010_NTA/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
    "&resultRecordCount=5000"
)


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def _normalize_2020(data: dict) -> dict:
    for feat in data.get("features", []):
        p = feat["properties"]
        feat["properties"] = {
            "boroname": p.get("BoroName", ""),
            "nta2020": p.get("NTA2020", ""),
            "ntaname": p.get("NTAName", ""),
        }
    return data


def _normalize_2010(data: dict) -> dict:
    for feat in data.get("features", []):
        p = feat["properties"]
        feat["properties"] = {
            "boroname": p.get("BoroName", ""),
            "nta": p.get("NTACode", ""),
            "ntaname": p.get("NTAName", ""),
        }
    return data


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)

    print("[fetch] 2020 NTAs (all NYC)...")
    d2020 = _normalize_2020(_fetch_json(URL_2020))
    p2020 = RAW / "nta2020_nyc.geojson"
    p2020.write_text(json.dumps(d2020), encoding="utf-8")
    print(f"[write] {p2020} ({len(d2020.get('features', []))} features)")

    print("[fetch] 2010 NTAs (all NYC)...")
    d2010 = _normalize_2010(_fetch_json(URL_2010_NYC))
    p2010 = RAW / "nta_nyc_2010.geojson"
    p2010.write_text(json.dumps(d2010), encoding="utf-8")
    print(f"[write] {p2010} ({len(d2010.get('features', []))} features)")

    # Backwards-compatible name for docs that cite nta.geojson (2020 layer)
    legacy = RAW / "nta.geojson"
    legacy.write_text(json.dumps(d2020), encoding="utf-8")
    print(f"[write] {legacy} (copy of 2020 layer for build_nta_features)")

    print(
        "\nNote: build_nta_features.load_manhattan_ntas() still needs the ACS GDB "
        "under data/raw/acs_nta_2014_2018/ unless you use nta_layers.load_nyc_ntas_for_zones()."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
