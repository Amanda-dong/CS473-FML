"""Orchestrator that runs all ETL modules and returns their outputs."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from src.data import (
    etl_311,
    etl_acs,
    etl_airbnb,
    etl_boundaries,
    etl_citibike,
    etl_inspections,
    etl_licenses,
    etl_permits,
    etl_pluto,
    etl_yelp,
)
from src.data.quality import validate_dataset_contract
from src.data.registry import DATASET_REGISTRY

logger = logging.getLogger(__name__)

_OPTIONAL_ENRICHMENT_DATASETS = frozenset({"yelp", "acs", "permits", "airbnb"})

_ETL_MODULES: dict[str, object] = {
    "permits": etl_permits,
    "licenses": etl_licenses,
    "inspections": etl_inspections,
    "acs": etl_acs,
    "pluto": etl_pluto,
    "citibike": etl_citibike,
    "airbnb": etl_airbnb,
    "yelp": etl_yelp,
    "complaints_311": etl_311,
    "boundaries": etl_boundaries,
}


def _run_module(module: object, limit: int) -> pd.DataFrame:
    """Execute the module's ETL entrypoint or placeholder fallback."""

    runner: Callable | None = getattr(module, "run_etl", None)
    if runner is None:
        runner = getattr(module, "run_placeholder_etl", None)
    if runner is None:
        raise AttributeError(
            f"{module!r} does not expose run_etl() or run_placeholder_etl()"
        )
    try:
        return runner(limit=limit)
    except TypeError:
        return runner()


def run_all_etl(
    limit: int = 2000,
    strict: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Run every ETL module and return results keyed by dataset name.

    Each module must expose a ``run_etl(limit)`` function.

    Parameters
    ----------
    limit : int
        Row limit passed to each ETL module.
    strict : bool
        If True, raise on first failure instead of continuing.

    Returns
    -------
    (results, status) where results maps dataset name -> DataFrame and
    status maps dataset name -> "ok", "empty", "skipped", or "failed".
    """
    results: dict[str, pd.DataFrame] = {}
    status: dict[str, str] = {}

    for name, module in _ETL_MODULES.items():
        spec = DATASET_REGISTRY.get(name, getattr(module, "DATASET_SPEC", None))
        try:
            if spec is not None and getattr(spec, "status", "") == "deprecated":
                status[name] = "skipped"
                logger.info("etl_runner: %s skipped (deprecated)", name)
                continue

            df = _run_module(module, limit=limit)
            if spec is None:
                raise KeyError(
                    f"{name} missing from DATASET_REGISTRY and module has no DATASET_SPEC"
                )
            validate_dataset_contract(df, spec)
            results[name] = df
            if df.empty:
                status[name] = "empty"
                logger.warning("etl_runner: %s returned 0 rows", name)
            else:
                status[name] = "ok"
                logger.info("etl_runner: %s returned %d rows", name, len(df))
        except (FileNotFoundError, RuntimeError) as exc:
            if name in _OPTIONAL_ENRICHMENT_DATASETS and spec is not None:
                logger.warning("etl_runner: %s skipped optional source: %s", name, exc)
                results[name] = pd.DataFrame(columns=list(spec.columns))
                status[name] = "skipped"
                continue
            if strict:
                raise
            logger.exception("etl_runner: %s failed", name)
            results[name] = pd.DataFrame()
            status[name] = "failed"
        except Exception:
            if strict:
                raise
            logger.exception("etl_runner: %s failed", name)
            results[name] = pd.DataFrame()
            status[name] = "failed"

    ok = sum(1 for v in status.values() if v == "ok")
    failed = sum(1 for v in status.values() if v == "failed")
    logger.info(
        "etl_runner: %d ok, %d failed, %d empty", ok, failed, len(status) - ok - failed
    )
    return results, status
