"""Central registry of source metadata."""

from .etl_311 import DATASET_SPEC as COMPLAINTS_311_SPEC
from .etl_acs import DATASET_SPEC as ACS_SPEC
from .etl_airbnb import DATASET_SPEC as AIRBNB_SPEC
from .etl_boundaries import DATASET_SPEC as BOUNDARIES_SPEC
from .etl_citibike import DATASET_SPEC as CITIBIKE_SPEC
from .etl_google_trends import DATASET_SPEC as GOOGLE_TRENDS_SPEC
from .etl_inspections import DATASET_SPEC as INSPECTIONS_SPEC
from .etl_licenses import DATASET_SPEC as LICENSES_SPEC
from .etl_permits import DATASET_SPEC as PERMITS_SPEC
from .etl_pluto import DATASET_SPEC as PLUTO_SPEC
from .etl_yelp import DATASET_SPEC as YELP_SPEC

DATASET_REGISTRY = {
    spec.name: spec
    for spec in (
        PERMITS_SPEC,
        LICENSES_SPEC,
        INSPECTIONS_SPEC,
        ACS_SPEC,
        PLUTO_SPEC,
        CITIBIKE_SPEC,
        AIRBNB_SPEC,
        YELP_SPEC,
        COMPLAINTS_311_SPEC,
        BOUNDARIES_SPEC,
        GOOGLE_TRENDS_SPEC,
    )
}
