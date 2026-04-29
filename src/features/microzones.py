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
    """Return well-known NYC zones covering all four microzone types."""

    return [
        # campus_walkshed
        MicrozoneDefinition(
            "tandon-campus", "campus_walkshed", "NYU Tandon / MetroTech", "frontend"
        ),
        MicrozoneDefinition(
            "bushwick-campus", "campus_walkshed", "Pratt Institute / Bushwick", "frontend"
        ),
        # lunch_corridor
        MicrozoneDefinition(
            "midtown-lunch", "lunch_corridor", "Midtown East Lunch Corridor", "frontend"
        ),
        MicrozoneDefinition(
            "flatiron-lunch", "lunch_corridor", "Flatiron / Madison Square Park", "frontend"
        ),
        # transit_catchment
        MicrozoneDefinition(
            "lic-transit", "transit_catchment", "Queens Plaza Transit Catchment", "data"
        ),
        MicrozoneDefinition(
            "fulton-transit", "transit_catchment", "Fulton St Transit Hub", "data"
        ),
        # business_district
        MicrozoneDefinition(
            "dumbo-biz", "business_district", "DUMBO / Brooklyn Tech Triangle", "data"
        ),
        MicrozoneDefinition(
            "hudson-yards-biz", "business_district", "Hudson Yards / West Chelsea", "frontend"
        ),
    ]
