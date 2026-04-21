"""Micro-zone definitions for campus and lunch-corridor recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicrozoneDefinition:
    """Metadata used by the frontend and geospatial workstreams."""

    zone_id: str
    zone_type: str
    label: str
    owner: str


def default_microzones() -> list[MicrozoneDefinition]:
    """Return example zones for local development and demo wiring."""

    return [
        MicrozoneDefinition(
            "tandon-campus", "campus_walkshed", "NYU Tandon / MetroTech", "frontend"
        ),
        MicrozoneDefinition(
            "midtown-lunch", "lunch_corridor", "Midtown East Lunch Corridor", "frontend"
        ),
        MicrozoneDefinition(
            "lic-transit", "transit_catchment", "Queens Plaza Transit Catchment", "data"
        ),
    ]
