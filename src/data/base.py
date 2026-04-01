"""Shared base classes for dataset placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Minimal source metadata for dataset-specific workstreams."""

    name: str
    owner: str
    spatial_unit: str
    time_grain: str
    description: str
    columns: tuple[str, ...]
    status: str = "planned"
    notes: str = ""


def build_empty_frame(spec: DatasetSpec) -> pd.DataFrame:
    """Return an empty frame with the columns expected for a source."""

    return pd.DataFrame(columns=list(spec.columns))


@dataclass
class BaseDatasetPipeline:
    """Placeholder ETL contract for source-specific modules."""

    spec: DatasetSpec
    tags: tuple[str, ...] = field(default_factory=tuple)

    def extract(self) -> pd.DataFrame:
        return build_empty_frame(self.spec)

    def transform(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        return raw_frame.copy()

    def load(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()
