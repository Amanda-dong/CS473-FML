"""Prompt and label scaffolding for Gemini-assisted review annotation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_CACHE_PATH = Path("data/processed/gemini_labels.parquet")


@dataclass(frozen=True)
class GeminiReviewLabel:
    """A single labeled review record from the offline annotation pass."""

    review_id: str
    sentiment: str
    concept_subtype: str
    confidence: float
    rationale: str


def build_label_prompt(review_text: str, subtype_candidates: tuple[str, ...]) -> str:
    """Return a consistent prompt template for the annotation workflow."""

    subtype_list = ", ".join(subtype_candidates)
    return (
        "Label the review for healthy-food demand and concept subtype.\n"
        f"Allowed subtypes: {subtype_list}.\n"
        f"Review: {review_text}"
    )


def _build_batch_prompt(
    review_texts: list[str], subtype_candidates: tuple[str, ...]
) -> str:
    """Build a prompt that labels multiple reviews in one API call."""
    subtype_list = ", ".join(subtype_candidates)
    lines = ["Label each review for healthy-food demand and concept subtype."]
    lines.append(f"Allowed subtypes: {subtype_list}.")
    lines.append(
        "Return a JSON array with one object per review, each having keys: "
        "sentiment, concept_subtype, confidence, rationale."
    )
    lines.append("")
    for i, text in enumerate(review_texts):
        lines.append(f"Review {i}: {text}")
    return "\n".join(lines)


def _cache_key(review_text: str, subtype_candidates: tuple[str, ...]) -> str:
    """Build a stable cache key from the review content and label taxonomy."""
    normalized_text = " ".join(str(review_text).split()).strip().lower()
    normalized_subtypes = "|".join(subtype_candidates)
    payload = f"{normalized_subtypes}\n{normalized_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache() -> dict[str, GeminiReviewLabel] | None:
    """Load cached labels from parquet if available."""
    if not _CACHE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_CACHE_PATH)
        if "rationale" not in df.columns:
            df["rationale"] = ""
        cache: dict[str, GeminiReviewLabel] = {
            str(rid): GeminiReviewLabel(
                review_id=str(rid),
                sentiment=str(sent),
                concept_subtype=str(sub),
                confidence=float(conf),
                rationale=str(rat),
            )
            for rid, sent, sub, conf, rat in zip(
                df["review_id"],
                df["sentiment"],
                df["concept_subtype"],
                df["confidence"],
                df["rationale"],
            )
        }
        return cache
    except Exception:
        return None


def _save_cache(labels: list[GeminiReviewLabel]) -> None:
    """Persist labels to parquet cache."""
    if not labels:
        return
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "review_id": l.review_id,
                "sentiment": l.sentiment,
                "concept_subtype": l.concept_subtype,
                "confidence": l.confidence,
                "rationale": l.rationale,
            }
            for l in labels
        ]
        df = pd.DataFrame(records)
        df.to_parquet(_CACHE_PATH, index=False)
    except Exception:
        pass


def label_reviews_with_gemini(
    reviews: list[str],
    subtypes: tuple[str, ...],
    api_key: str | None = None,
) -> list[GeminiReviewLabel]:
    """Label a list of review texts using Gemini. Raises if API key missing.

    Parameters
    ----------
    reviews:
        List of review text strings.
    subtypes:
        Allowed concept subtype labels.
    api_key:
        Gemini API key. Falls back to GEMINI_API_KEY env var if None.

    Returns
    -------
    List of GeminiReviewLabel with one entry per review.
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not resolved_key:
        raise RuntimeError(
            "GEMINI_API_KEY env var required for review labeling. "
            "No synthetic fallback — real labels only."
        )

    try:
        import google.genai as genai  # type: ignore[import]
    except ImportError:
        raise ImportError("google-genai package required: pip install google-genai")

    # Check cache
    cache = _load_cache() or {}
    labels: list[GeminiReviewLabel] = [None] * len(reviews)  # type: ignore[list-item]
    uncached_indices: list[int] = []

    for i, _review in enumerate(reviews):
        review_id = _cache_key(reviews[i], subtypes)
        if review_id in cache:
            labels[i] = cache[review_id]
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return labels

    client = genai.Client(api_key=resolved_key)

    # Batch in groups of 10
    batch_size = 10
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_indices = uncached_indices[batch_start : batch_start + batch_size]
        batch_texts = [reviews[i] for i in batch_indices]

        try:
            prompt = _build_batch_prompt(batch_texts, subtypes)
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw = response.text or "[]"
            data_list = json.loads(raw)
            if not isinstance(data_list, list):
                data_list = [data_list]

            for rel_idx, abs_idx in enumerate(batch_indices):
                review_id = _cache_key(reviews[abs_idx], subtypes)
                if rel_idx < len(data_list):
                    data = data_list[rel_idx]
                    label = GeminiReviewLabel(
                        review_id=review_id,
                        sentiment=str(data.get("sentiment", "neutral")),
                        concept_subtype=str(
                            data.get(
                                "concept_subtype",
                                subtypes[0] if subtypes else "unknown",
                            )
                        ),
                        confidence=float(data.get("confidence", 0.85)),
                        rationale=str(data.get("rationale", "")),
                    )
                else:
                    raise RuntimeError(
                        f"Gemini returned fewer labels ({len(data_list)}) "
                        f"than reviews in batch ({len(batch_indices)})"
                    )
                labels[abs_idx] = label
                cache[review_id] = label
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Gemini labeling failed for batch starting at index {batch_start}: {exc}"
            ) from exc

    # Save all to cache
    _save_cache(list(cache.values()))

    return labels
