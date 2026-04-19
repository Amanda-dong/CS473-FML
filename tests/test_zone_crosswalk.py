"""Tests for NTA → micro-zone resolution."""

from __future__ import annotations

from src.features.zone_crosswalk import resolve_nta_to_zone_id


def test_resolve_single_nta_maps_to_one_zone() -> None:
    assert resolve_nta_to_zone_id("MN22") == "mn-nyu-wash-sq"


def test_resolve_ambiguous_nta_uses_primary() -> None:
    assert resolve_nta_to_zone_id("MN17") == "mn-midtown-e"


def test_resolve_unknown_nta_returns_none() -> None:
    assert resolve_nta_to_zone_id("MN99") is None
    assert resolve_nta_to_zone_id("") is None
    assert resolve_nta_to_zone_id(None) is None
