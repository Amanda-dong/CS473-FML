"""Boilerplate for the neighborhood feature matrix workstream."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureTable:
    """Metadata for feature tables owned by separate team members."""

    name: str
    owner: str
    join_keys: tuple[str, ...]


def build_feature_matrix(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join a dictionary of source feature tables on shared keys."""

    tables = [frame for frame in feature_tables.values() if not frame.empty]
    if not tables:
        return pd.DataFrame(columns=["zone_id", "time_key"])

    merged = tables[0].copy()
    for frame in tables[1:]:
        join_keys = [column for column in ("zone_id", "time_key") if column in merged.columns and column in frame.columns]
        merged = merged.merge(frame, how="outer", on=join_keys)
    return merged
