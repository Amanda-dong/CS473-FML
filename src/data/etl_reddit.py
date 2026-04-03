"""ETL for Reddit mention signals via PRAW."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

DATASET_SPEC = DatasetSpec(
    name="reddit",
    owner="nlp",
    spatial_unit="community_district",
    time_grain="month",
    description="Coarse-grained social mention signals for food interest and buzz.",
    columns=("month", "community_district", "mention_text", "subreddit"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


# ---------------------------------------------------------------------------
# NTA / community-district name fragments used for text matching
# ---------------------------------------------------------------------------

_NTA_NAMES = [
    "Brooklyn", "Manhattan", "Queens", "Bronx", "Harlem",
    "Astoria", "Flushing", "Williamsburg", "Bushwick", "Flatbush",
    "Greenpoint", "Sunset Park", "Jackson Heights", "Bay Ridge", "Ridgewood",
]

_SUBREDDITS = ["nyc", "AskNYC"]


def _extract_community_district(text: str) -> str:
    """Return the first matched NTA name fragment or 'Unknown'."""
    for name in _NTA_NAMES:
        if name.lower() in text.lower():
            return name
    return "Unknown"


def fetch() -> pd.DataFrame:
    """Fetch posts from r/nyc and r/AskNYC using PRAW."""
    import praw  # type: ignore[import]

    client_id = os.environ["REDDIT_CLIENT_ID"]
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "nyc-restaurant-intel/1.0")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=180)
    rows = []
    for sub_name in _SUBREDDITS:
        sub = reddit.subreddit(sub_name)
        for post in sub.search("healthy food restaurant", sort="new", limit=500):
            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if created < cutoff:
                continue
            text = f"{post.title} {post.selftext}"
            rows.append({
                "month": created.strftime("%Y-%m"),
                "community_district": _extract_community_district(text),
                "mention_text": post.title[:200],
                "subreddit": sub_name,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(DATASET_SPEC.columns))


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df[list(DATASET_SPEC.columns)].reset_index(drop=True)


def run_etl(limit: int = 50000) -> pd.DataFrame:  # noqa: ARG001
    """Fetch real Reddit data via PRAW. Raises if credentials missing or fetch fails."""
    if not os.environ.get("REDDIT_CLIENT_ID"):
        raise RuntimeError(
            "etl_reddit: REDDIT_CLIENT_ID env var required. "
            "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT."
        )
    raw = fetch()
    return transform(raw)
