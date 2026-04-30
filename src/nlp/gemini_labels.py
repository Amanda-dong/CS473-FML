"""Prompt and label scaffolding for Gemini-assisted review annotation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_CACHE_PATH = Path("data/processed/gemini_labels.parquet")
_HALAL_RELEVANCE_LABELS = ("explicit_halal", "implicit_halal", "not_related")
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
_DEFAULT_PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
_DEFAULT_PORTKEY_MODEL = "@vertexai/gemini-2.5-flash-lite"
_LABEL_SCHEMA_VERSION = "healthy_food_v1"


@dataclass(frozen=True)
class GeminiReviewLabel:
    """A single labeled review record from the offline annotation pass."""

    review_id: str
    sentiment: str
    halal_relevance: (
        str  # Note: Retained as 'halal_relevance' for data schema compatibility.
    )
    concept_subtype: str
    confidence: float
    rationale: str = ""


def build_label_prompt(review_text: str, subtype_candidates: tuple[str, ...]) -> str:
    """Return a consistent prompt template for the annotation workflow."""

    subtype_list = ", ".join(subtype_candidates)
    return (
        "Label this Yelp review for healthy food demand analysis.\n"
        "Return JSON only with keys: sentiment, halal_relevance, "
        "concept_subtype, confidence.\n"
        "sentiment must be one of: positive, neutral, negative.\n"
        "halal_relevance must be one of: explicit_halal, implicit_halal, "
        "not_related.\n"
        "The review text may include business context such as business name "
        "and categories. Use that context when judging healthy food relevance.\n"
        "Note: halal is one of many healthy food categories. concept_subtype captures the specific food category.\n"
        "Use explicit_halal when the review text, business name, or business "
        "categories clearly mention halal.\n"
        "Use implicit_halal only when the review implies demand for halal options without explicit mention.\n"
        "Use not_related when halal-specific demand is not clear. Be conservative and "
        "do not guess.\n"
        f"Allowed subtypes: {subtype_list}.\n"
        "Do not create new concept_subtype labels. If none fit, use other.\n"
        f"Review: {review_text}"
    )


def _build_batch_prompt(
    review_texts: list[str], subtype_candidates: tuple[str, ...]
) -> str:
    """Build a prompt that labels multiple reviews in one API call."""
    subtype_list = ", ".join(subtype_candidates)
    lines = ["Label each Yelp review for healthy food demand analysis."]
    lines.append("Return JSON only.")
    lines.append("Return a JSON array with one object per review.")
    lines.append(
        "Each object must have keys: sentiment, halal_relevance, "
        "concept_subtype, confidence."
    )
    lines.append("sentiment must be one of: positive, neutral, negative.")
    lines.append(
        "halal_relevance must be one of: explicit_halal, implicit_halal, not_related."
    )
    lines.append(
        "Definitions (Note: Halal is one of many healthy food subtypes. concept_subtype will identify the category):"
    )
    lines.append(
        "- explicit_halal: the review text, business name, or business "
        "categories clearly mention halal."
    )
    lines.append(
        "- implicit_halal: the review implies halal demand, lack of halal "
        "options, or discusses a known halal concept without saying halal."
    )
    lines.append("- not_related: halal-specific demand is not clear from the review.")
    lines.append(
        "Each review may include business name and categories before the review "
        "text. Use that business context when judging healthy food relevance."
    )
    lines.append("Be conservative. Do not infer demand from positive sentiment alone.")
    lines.append("confidence must be a number from 0.0 to 1.0, not a word.")
    lines.append(f"Allowed subtypes: {subtype_list}.")
    lines.append("Do not create new concept_subtype labels. If none fit, use other.")
    lines.append("")
    for i, text in enumerate(review_texts):
        lines.append(f"Review {i}: {text}")
    return "\n".join(lines)


def _cache_key(review_text: str, subtype_candidates: tuple[str, ...]) -> str:
    """Build a stable cache key from the review content and label taxonomy."""
    normalized_text = " ".join(str(review_text).split()).strip().lower()
    normalized_subtypes = "|".join(subtype_candidates)
    payload = (
        f"{_LABEL_SCHEMA_VERSION}\n{normalized_subtypes}\n{normalized_text}".encode(
            "utf-8"
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _load_cache() -> dict[str, GeminiReviewLabel] | None:
    """Load cached labels from parquet if available."""
    if not _CACHE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_CACHE_PATH)
        if "halal_relevance" not in df.columns:
            df["halal_relevance"] = "not_related"
        if "rationale" not in df.columns:
            df["rationale"] = ""
        cache: dict[str, GeminiReviewLabel] = {
            str(rid): GeminiReviewLabel(
                review_id=str(rid),
                sentiment=str(sent),
                halal_relevance=str(rel),
                concept_subtype=str(sub),
                confidence=float(conf),
                rationale=str(rat),
            )
            for rid, sent, rel, sub, conf, rat in zip(
                df["review_id"],
                df["sentiment"],
                df["halal_relevance"],
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
                "review_id": label.review_id,
                "sentiment": label.sentiment,
                "halal_relevance": label.halal_relevance,
                "concept_subtype": label.concept_subtype,
                "confidence": label.confidence,
                "rationale": label.rationale,
            }
            for label in labels
        ]
        df = pd.DataFrame(records)
        df.to_parquet(_CACHE_PATH, index=False)
    except Exception:
        pass


def _parse_label_payload(raw: str) -> list[dict]:
    """Parse model JSON output, tolerating accidental markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    data_list = json.loads(cleaned or "[]")
    if not isinstance(data_list, list):
        data_list = [data_list]
    return data_list


def _generate_label_payload(prompt: str, api_key: str) -> list[dict]:
    """Call the configured labeling model through Portkey or Google GenAI."""
    portkey_key = os.environ.get("PORTKEY_API_KEY", "").strip()
    if portkey_key:
        try:
            from portkey_ai import Portkey  # type: ignore[import]
        except ImportError:
            raise ImportError("portkey-ai package required: pip install portkey-ai")

        client = Portkey(
            base_url=os.environ.get("PORTKEY_BASE_URL", _DEFAULT_PORTKEY_BASE_URL),
            api_key=portkey_key,
        )
        response = client.chat.completions.create(
            model=os.environ.get("PORTKEY_MODEL", _DEFAULT_PORTKEY_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": "You label Yelp reviews and return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or "[]"
        return _parse_label_payload(raw)

    try:
        import google.genai as genai  # type: ignore[import]
    except ImportError:
        raise ImportError("google-genai package required: pip install google-genai")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL),
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw = response.text or "[]"
    return _parse_label_payload(raw)


def _coerce_confidence(value: object, default: float = 0.85) -> float:
    """Convert model confidence output to a bounded float."""
    if isinstance(value, str):
        mapped = {"high": 0.9, "medium": 0.7, "low": 0.4}
        normalized = value.strip().lower()
        if normalized in mapped:
            return mapped[normalized]
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _coerce_subtype(value: object, subtypes: tuple[str, ...]) -> str:
    """Normalize model subtype output to the allowed taxonomy."""
    fallback = subtypes[0] if subtypes else "unknown"
    subtype = str(value or fallback).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "food_truck": "street_food",
        "food_trucks": "street_food",
        "cart": "street_food",
        "halal_cart": "street_food",
        "restaurant": "fast_casual",
    }
    subtype = aliases.get(subtype, subtype)
    if subtype in subtypes:
        return subtype
    if "other" in subtypes:
        return "other"
    return fallback


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
    resolved_key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("PORTKEY_API_KEY")
    )

    if not resolved_key:
        raise RuntimeError(
            "GEMINI_API_KEY or PORTKEY_API_KEY env var required for review labeling. "
            "No synthetic fallback — real labels only."
        )

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

    # Batch in groups of 10
    batch_size = 10
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_indices = uncached_indices[batch_start : batch_start + batch_size]
        batch_texts = [reviews[i] for i in batch_indices]

        try:
            prompt = _build_batch_prompt(batch_texts, subtypes)
            data_list = _generate_label_payload(prompt, resolved_key)

            for rel_idx, abs_idx in enumerate(batch_indices):
                review_id = _cache_key(reviews[abs_idx], subtypes)
                if rel_idx < len(data_list):
                    data = data_list[rel_idx]
                    halal_relevance = str(data.get("halal_relevance", "not_related"))
                    if halal_relevance not in _HALAL_RELEVANCE_LABELS:
                        halal_relevance = "not_related"
                    label = GeminiReviewLabel(
                        review_id=review_id,
                        sentiment=str(data.get("sentiment", "neutral")),
                        halal_relevance=halal_relevance,
                        concept_subtype=_coerce_subtype(
                            data.get("concept_subtype"), subtypes
                        ),
                        confidence=_coerce_confidence(data.get("confidence", 0.85)),
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
