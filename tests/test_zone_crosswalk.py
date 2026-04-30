"""Tests for NTA → micro-zone resolution."""

from __future__ import annotations

from src.features.zone_crosswalk import resolve_nta_to_zone_id


def test_resolve_single_nta_maps_to_one_zone() -> None:
    # MN0202 = Greenwich Village (2020 NTA code) → mn-nyu-wash-sq
    assert resolve_nta_to_zone_id("MN0202") == "mn-nyu-wash-sq"


def test_resolve_ambiguous_nta_uses_primary() -> None:
    # MN0604 is shared by mn-midtown-e and mn-lic-adj; primary is mn-midtown-e
    assert resolve_nta_to_zone_id("MN0604") == "mn-midtown-e"


def test_resolve_unknown_nta_returns_none() -> None:
    assert resolve_nta_to_zone_id("MN99") is None
    assert resolve_nta_to_zone_id("") is None
    assert resolve_nta_to_zone_id(None) is None
