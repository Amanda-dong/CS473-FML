"""Tests for NTA → micro-zone resolution."""

from __future__ import annotations

from src.features.zone_crosswalk import resolve_nta_to_zone_id, _load_all_nta_codes


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


def test_load_all_nta_codes_fallback(monkeypatch, tmp_path) -> None:
    # Mock Path to point to a non-existent or malformed file

    # First test: no files exist
    monkeypatch.setattr(
        "src.features.zone_crosswalk.Path", lambda x: tmp_path / "nonexistent"
    )
    assert _load_all_nta_codes() == []

    # Second test: malformed JSON
    malformed = tmp_path / "malformed.geojson"
    malformed.write_text("not json")
    monkeypatch.setattr("src.features.zone_crosswalk.Path", lambda x: malformed)
    assert _load_all_nta_codes() == []
