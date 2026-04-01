"""Tests for pipeline orchestration helpers."""

from src.pipeline.orchestrator import ProjectPipeline
from src.pipeline.stages import PIPELINE_STAGES


def test_pipeline_tracks_known_stages() -> None:
    """The pipeline should accept documented stage names."""

    pipeline = ProjectPipeline()
    pipeline.run_stage(PIPELINE_STAGES[0])
    assert PIPELINE_STAGES[0] in pipeline.completed_stages
