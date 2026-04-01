"""Prompt and label scaffolding for Gemini-assisted review annotation."""

from __future__ import annotations

from dataclasses import dataclass


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
