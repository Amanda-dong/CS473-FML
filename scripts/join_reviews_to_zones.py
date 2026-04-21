"""Join Yelp Fusion reviews with ``yelp_business_zones`` for zone_id + time_key.

Adds:

- ``time_key``: calendar year from ``review_date`` (int).
- ``review_id``: stable SHA-256 hex id from (restaurant_id, review_date, review_text).
- ``nta``, ``zone_id``, ``in_nyc_nta``, ``in_modeled_microzone`` from the zones table.

Usage (from repo root)::

    python scripts/join_reviews_to_zones.py

    python scripts/join_reviews_to_zones.py --reviews data/raw/yelp_reviews_fusion_2022_2026.csv \\
        --zones data/processed/yelp_business_zones.csv \\
        --output data/processed/yelp_reviews_with_zones.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
DEFAULT_ZONES = PROC / "yelp_business_zones.csv"
DEFAULT_OUTPUT = PROC / "yelp_reviews_with_zones.csv"


def _default_reviews_path() -> Path:
    filtered = RAW / "yelp_reviews_fusion_2022_2026.csv"
    if filtered.is_file():
        return filtered
    return RAW / "yelp_reviews_fusion.csv"


def _stable_review_id(restaurant_id: str, review_date: str, review_text: str) -> str:
    payload = f"{restaurant_id}\0{review_date}\0{review_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reviews", type=Path, default=None, help="Yelp Fusion reviews CSV."
    )
    p.add_argument(
        "--zones",
        type=Path,
        default=DEFAULT_ZONES,
        help="Output of assign_yelp_business_zones.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    import pandas as pd

    args = _parse_args()
    reviews_path = args.reviews if args.reviews is not None else _default_reviews_path()

    if not reviews_path.is_file():
        print(f"error: reviews file not found: {reviews_path}", file=sys.stderr)
        return 1
    if not args.zones.is_file():
        print(f"error: zones file not found: {args.zones}", file=sys.stderr)
        return 1

    rev = pd.read_csv(reviews_path)
    if "restaurant_id" not in rev.columns and "business_id" in rev.columns:
        rev["restaurant_id"] = rev["business_id"]
    if "restaurant_id" not in rev.columns:
        print(
            "error: need restaurant_id or business_id column in reviews CSV",
            file=sys.stderr,
        )
        return 1

    zones = pd.read_csv(args.zones)
    if "restaurant_id" not in zones.columns:
        print("error: zones CSV must have restaurant_id", file=sys.stderr)
        return 1

    rev["restaurant_id"] = rev["restaurant_id"].astype(str).str.strip()
    zones["restaurant_id"] = zones["restaurant_id"].astype(str).str.strip()

    zone_cols = [
        c
        for c in ("nta", "zone_id", "in_nyc_nta", "in_modeled_microzone")
        if c in zones.columns
    ]
    z = zones[["restaurant_id", *zone_cols]].drop_duplicates(subset=["restaurant_id"])

    out = rev.merge(z, on="restaurant_id", how="left")

    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")
    out["time_key"] = out["review_date"].dt.year
    out["review_date"] = out["review_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    texts = out["review_text"].fillna("").astype(str)
    rd = out["review_date"].astype(str)
    rid = out["restaurant_id"].astype(str)
    out["review_id"] = [_stable_review_id(a, b, c) for a, b, c in zip(rid, rd, texts)]

    # Sensible column order
    front = [
        "review_id",
        "restaurant_id",
        "business_id",
        "review_date",
        "time_key",
        "zone_id",
        "nta",
        "in_nyc_nta",
        "in_modeled_microzone",
        "rating",
        "review_text",
    ]
    ordered = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    out = out[ordered + rest]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    n = len(out)
    z_ok = int(out["zone_id"].notna().sum()) if "zone_id" in out.columns else 0
    print(f"[write] {args.output}")
    print(f"[stats] rows={n} with zone_id={z_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
