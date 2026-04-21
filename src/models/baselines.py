"""Shared baseline bundles for placeholder experimentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineRun:
    """A lightweight record describing an experiment slot."""

    name: str
    owner: str
    notes: str


def build_baseline_runs() -> list[BaselineRun]:
    """Return the initial model workstreams for the team."""

    return [
        BaselineRun("trajectory_kmeans", "ml", "Unsupervised neighborhood regime discovery."),
        BaselineRun("survival_cox", "ml", "Right-censored restaurant survival baseline."),
        BaselineRun("healthy_gap_score", "ml", "Interpretable weighted recommendation baseline."),
    ]
