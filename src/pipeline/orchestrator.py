"""A lightweight orchestrator that documents the intended workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from .stages import PIPELINE_STAGES


@dataclass
class ProjectPipeline:
    """Track the ordered stages of the current scaffold."""

    completed_stages: list[str] = field(default_factory=list)

    def run_stage(self, stage_name: str) -> None:
        if stage_name not in PIPELINE_STAGES:
            raise ValueError(f"Unknown stage: {stage_name}")
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
