"""Load Yelp + Gemini labeled reviews for Streamlit neighborhood evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REL_ORDER = {"explicit_halal": 0, "implicit_halal": 1, "not_related": 2}


def _venue_key_column(df: pd.DataFrame) -> pd.Series:
    """Stable key per venue: restaurant_id when present, else normalized business_name, else synthetic id."""
    if "restaurant_id" in df.columns:
        rid = df["restaurant_id"].astype(str).str.strip()
        rid = rid.replace({"nan": "", "none": "", "<na>": ""})
        bad_rid = rid.str.lower().isin(["", "nan", "none"])
        rid = rid.where(~bad_rid, "")
    else:
        rid = pd.Series("", index=df.index, dtype=str)
    if "business_name" in df.columns:
        name = df["business_name"].fillna("").astype(str).str.strip().str.lower()
    else:
        name = pd.Series("", index=df.index, dtype=str)
    key = rid.where(rid != "", name)
    missing = (key == "") | (key.str.lower() == "nan")
    key = key.where(~missing, "anon_" + df.index.astype(str))
    return key


def evidence_csv_path(repo_root: Path) -> Path:
    return repo_root / "data" / "raw" / "gemini_labels_full.csv"


def load_labeled_reviews(repo_root: Path) -> pd.DataFrame | None:
    """Returns a normalized dataframe for lookups, or None if file missing."""
    path = evidence_csv_path(repo_root)
    if not path.is_file():
        return None

    usecols_candidates = [
        "nta",
        "review_text",
        "rating",
        "halal_relevance",
        "business_name",
        "review_date",
        "restaurant_id",
    ]
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in usecols_candidates if c in all_cols]

    df = pd.read_csv(path, usecols=usecols)
    if "nta" not in df.columns:
        return None

    df = df.dropna(subset=["nta"]).copy()
    df["nta_norm"] = df["nta"].astype(str).str.strip().str.upper()
    df["review_text"] = df["review_text"].fillna("").astype(str)
    if "halal_relevance" in df.columns:
        df["halal_relevance"] = (
            df["halal_relevance"].fillna("not_related").astype(str).str.strip()
        )
    else:
        df["halal_relevance"] = "not_related"

    df["_prio"] = df["halal_relevance"].map(_REL_ORDER).fillna(3)
    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
    df["_venue_key"] = _venue_key_column(df)
    return df


def nta_review_counts(pool: pd.DataFrame, nta_id: str) -> dict[str, int]:
    nid = str(nta_id).strip().upper()
    sub = pool[pool["nta_norm"] == nid]
    rel_norm = sub["halal_relevance"].astype(str).str.strip().str.lower()
    uniq = (
        int(sub["_venue_key"].nunique())
        if not sub.empty and "_venue_key" in sub.columns
        else 0
    )
    explicit_halal = int((rel_norm == "explicit_halal").sum())
    implicit_halal = int((rel_norm == "implicit_halal").sum())
    not_related = int((rel_norm == "not_related").sum())
    accounted = explicit_halal + implicit_halal + not_related
    other_labels = max(0, len(sub) - accounted)
    return {
        "total": len(sub),
        "explicit_halal": explicit_halal,
        "implicit_halal": implicit_halal,
        "not_related": not_related,
        "other_labels": other_labels,
        "unique_venues": uniq,
    }


def sample_reviews_for_nta(pool: pd.DataFrame, nta_id: str, k: int = 6) -> pd.DataFrame:
    """Prefer explicit_halal → implicit_halal; then rating desc; at most one row per venue."""
    nid = str(nta_id).strip().upper()
    sub = pool[pool["nta_norm"] == nid].copy()
    if sub.empty:
        return sub

    if "_venue_key" not in sub.columns:
        sub["_venue_key"] = _venue_key_column(sub)

    sub = sub.sort_values(["_prio", "rating"], ascending=[True, False])
    sub = sub.drop_duplicates(subset=["_venue_key"], keep="first")
    return sub.head(k).drop(columns=["_prio", "_venue_key"], errors="ignore")


def clip_review(text: str, max_chars: int = 360) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
