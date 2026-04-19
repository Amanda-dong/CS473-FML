"""Filter Yelp Fusion review CSV by calendar year (local file, no API).

The CSV from ``fetch_yelp_reviews_fusion_v2.py`` can include many years (e.g. 2018).
This script **does not modify** that file by default: it reads it and writes a
**separate** file with only rows whose ``review_date`` falls in the chosen range
(default **2022–2026**).

Usage::

    python scripts/filter_yelp_reviews_fusion.py

    # If ``python`` is not on PATH (Windows), try: ``py scripts/...``

    python scripts/filter_yelp_reviews_fusion.py --input data/raw/yelp_reviews_fusion.csv ^
        --output data/raw/yelp_reviews_fusion_2022_2026.csv --min-year 2022 --max-year 2026

To train or ETL on the filtered file, point ``YELP_DATA_PATH`` at the output path
or swap filenames as your team prefers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
DEFAULT_INPUT = RAW / "yelp_reviews_fusion.csv"
DEFAULT_OUTPUT = RAW / "yelp_reviews_fusion_2022_2026.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Full API review CSV (all years).")
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Filtered CSV (default: yelp_reviews_fusion_2022_2026.csv).",
    )
    p.add_argument("--min-year", type=int, default=2022)
    p.add_argument("--max-year", type=int, default=2026)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    min_y, max_y = args.min_year, args.max_year
    if min_y > max_y:
        print("error: --min-year must be <= --max-year", file=sys.stderr)
        return 1

    src = args.input
    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1

    df = pd.read_csv(src)
    if "review_date" not in df.columns:
        print("error: column 'review_date' missing", file=sys.stderr)
        return 1

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["review_date"])
    years = df["review_date"].dt.year
    df = df[(years >= min_y) & (years <= max_y)].copy()
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    n_after = len(df)
    print(f"[filter] {src.name}: rows {n_before} -> {n_after} (years {min_y}-{max_y})")
    print(f"[write]  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
