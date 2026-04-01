"""Simple placeholder logic for location mentions in text."""

from __future__ import annotations


def extract_location_mentions(text: str, candidates: tuple[str, ...]) -> list[str]:
    """Return candidate location strings that appear in the input text."""

    lowered = text.lower()
    return [candidate for candidate in candidates if candidate.lower() in lowered]
