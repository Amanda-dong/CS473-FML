"""Lightweight subtype classification helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.utils.taxonomy import healthy_taxonomy


def batch_classify(texts: list[str]) -> list[str]:
    """Classify a list of texts into subtypes.

    Parameters
    ----------
    texts:
        List of free-text strings to classify.

    Returns
    -------
    List of subtype strings, one per input text.
    """
    return [classify_subtype_keyword(text) for text in texts]


def classify_subtype_keyword(text: str) -> str:
    """Map free text to the first matching healthy subtype keyword set."""

    lowered = text.lower()
    for subtype, keywords in healthy_taxonomy().items():
        if any(keyword in lowered for keyword in keywords):
            return subtype
    return "unknown"


# Keep backward-compatible alias
classify_subtype = classify_subtype_keyword


def classify_subtype_embedding(
    text: str,
    subtype_centroids: dict[str, np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> str:
    """Classify text by cosine similarity to subtype centroids.

    Parameters
    ----------
    text:
        Free-text string to classify.
    subtype_centroids:
        Mapping of subtype name to centroid embedding vector.
    embed_fn:
        Function that takes a list of strings and returns (N, D) embedding array.

    Returns
    -------
    Best-matching subtype name, or "unknown" if no centroids provided.
    """
    if not subtype_centroids:
        return "unknown"

    embedding = embed_fn([text])[0]
    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        return "unknown"
    embedding = embedding / emb_norm

    best_subtype = "unknown"
    best_sim = -1.0

    for subtype, centroid in subtype_centroids.items():
        c_norm = np.linalg.norm(centroid)
        if c_norm == 0:
            continue
        sim = float(np.dot(embedding, centroid / c_norm))
        if sim > best_sim:
            best_sim = sim
            best_subtype = subtype

    return best_subtype
