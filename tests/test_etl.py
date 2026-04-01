"""Tests for dataset scaffold modules."""

from src.data.audit import build_default_audit_rows
from src.data.registry import DATASET_REGISTRY


def test_registry_contains_core_datasets() -> None:
    """The scaffold should expose the main active sources."""

    for dataset_name in ("permits", "licenses", "inspections", "acs", "pluto"):
        assert dataset_name in DATASET_REGISTRY


def test_audit_rows_cover_the_registry() -> None:
    """The audit helper should expose one row per registered dataset."""

    assert len(build_default_audit_rows()) == len(DATASET_REGISTRY)
