"""Assign each Yelp business to an NTA (NYC polygons) and a micro-zone id.

Requires **2010 NYC NTA** polygons (ACS codes like MN22, BK09):

- Run ``python scripts/download_nta_geojson.py`` once to create
  ``data/raw/nta_nyc_2010.geojson`` (and optional ``nta.geojson`` for Phase1 ETL).

Uses ``src.data.nta_layers.load_nyc_ntas_for_zones`` — **no** ACS GDB needed
for this script (unlike ``build_nta_features.load_manhattan_ntas``).

Yelp rows whose coordinates fall **outside** NYC NTAs get empty ``nta`` /
``zone_id``. Rows inside an NTA that is not in ``ZONE_TO_NTA`` get ``nta``
filled and ``zone_id`` empty (``in_modeled_microzone`` False).

Usage (from repo root)::

    python scripts/assign_yelp_business_zones.py
    python scripts/assign_yelp_business_zones.py --input data/raw/yelp_business.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_INPUT = REPO / "data" / "raw" / "yelp_business.csv"
DEFAULT_OUTPUT = REPO / "data" / "processed" / "yelp_business_zones.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    import pandas as pd

    from src.data.nta_layers import load_nyc_ntas_for_zones
    from src.features.yelp_microzones import assign_yelp_business_zones

    args = _parse_args()
    if not args.input.is_file():
        print(f"error: missing {args.input}", file=sys.stderr)
        return 1

    try:
        nta_gdf = load_nyc_ntas_for_zones()
    except Exception as exc:  # noqa: BLE001
        print(
            "error: could not load NYC NTA GeoJSON. "
            "Run: python scripts/download_nta_geojson.py",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    yelp = pd.read_csv(args.input)
    out = assign_yelp_business_zones(yelp, nta_gdf)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    n = len(out)
    m = int(out["in_nyc_nta"].sum()) if "in_nyc_nta" in out.columns else 0
    z = int(out["in_modeled_microzone"].sum()) if "in_modeled_microzone" in out.columns else 0
    print(f"[write] {args.output}")
    print(f"[stats] businesses={n} in_nyc_nta={m} in_modeled_microzone={z}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
