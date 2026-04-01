"""Lightweight subtype classification helpers."""

from __future__ import annotations

from src.utils.taxonomy import healthy_taxonomy


def classify_subtype(text: str) -> str:
    """Map free text to the first matching healthy subtype keyword set."""

    lowered = text.lower()
    for subtype, keywords in healthy_taxonomy().items():
        if any(keyword in lowered for keyword in keywords):
            return subtype
    return "unknown"
