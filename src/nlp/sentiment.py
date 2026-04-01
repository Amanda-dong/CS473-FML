"""Placeholder sentiment helpers for the review-labeling workstream."""


def allowed_sentiment_labels() -> tuple[str, ...]:
    """Return the compact sentiment label space for offline annotation."""

    return ("positive", "neutral", "negative")
