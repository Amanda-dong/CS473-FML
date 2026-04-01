"""Default audit rows for the active data sources."""

from __future__ import annotations

from src.schemas.datasets import DatasetAuditRow

from .registry import DATASET_REGISTRY


def build_default_audit_rows() -> list[DatasetAuditRow]:
    """Return one audit row per registered dataset."""

    return [
        DatasetAuditRow(
            name=spec.name,
            owner=spec.owner,
            spatial_unit=spec.spatial_unit,
            time_grain=spec.time_grain,
            status=spec.status,
            notes=spec.notes,
        )
        for spec in DATASET_REGISTRY.values()
    ]
