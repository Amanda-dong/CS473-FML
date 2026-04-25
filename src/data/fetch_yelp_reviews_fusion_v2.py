from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_BASE = "https://api.yelp.com/v3"
DEFAULT_INPUT = Path("data/raw/yelp_business.csv")
DEFAULT_OUTPUT = Path("data/raw/yelp_reviews_fusion.csv")


def _load_repo_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)


def _api_key(cli_key: str | None) -> str:
    key = (cli_key or os.environ.get("YELP_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("Missing YELP_API_KEY (or pass --api-key).")
    return key


def _business_ids(path: Path, id_column: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in {path}")
    ids = df[id_column].dropna().astype(str).str.strip()
    return ids[ids != ""].drop_duplicates().tolist()


def _fetch_one(
    session: requests.Session,
    business_id: str,
    per_business_limit: int,
) -> list[dict[str, object]]:
    params = {"limit": per_business_limit, "date_from": "2022", "date_to": "2025"}
    resp = session.get(f"{API_BASE}/businesses/{business_id}/reviews", params=params, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    reviews = resp.json().get("reviews", [])[:per_business_limit]
    return [
        {
            "review_date": r.get("time_created", ""),
            "business_id": business_id,
            "restaurant_id": business_id,
            "rating": r.get("rating"),
            "review_text": r.get("text", ""),
        }
        for r in reviews
    ]


def fetch_reviews(
    input_csv: Path,
    output_csv: Path,
    api_key: str | None,
    id_column: str,
    per_business_limit: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    if per_business_limit <= 0:
        raise ValueError("--per-business-limit must be > 0")
    ids = _business_ids(input_csv, id_column=id_column)

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_api_key(api_key)}"

    rows: list[dict[str, object]] = []
    total = len(ids)
    for i, bid in enumerate(ids, start=1):
        try:
            rows.extend(_fetch_one(session, bid, per_business_limit))
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "unknown"
            print(f"[warn] {bid} HTTP {code}")
        except requests.RequestException as exc:
            print(f"[warn] {bid} request error: {exc}")

        if i % 100 == 0 or i == total:
            print(f"[progress] {i}/{total}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    out = pd.DataFrame(
        rows,
        columns=["review_date", "business_id", "restaurant_id", "rating", "review_text"],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out


def main() -> None:
    _load_repo_env()
    parser = argparse.ArgumentParser(description="Fetch Yelp Fusion reviews from business IDs.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--per-business-limit", type=int, default=7)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    df = fetch_reviews(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        api_key=args.api_key,
        id_column=args.id_column,
        per_business_limit=args.per_business_limit,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"Saved {len(df)} reviews to {args.output_csv}")


if __name__ == "__main__":
    main()
