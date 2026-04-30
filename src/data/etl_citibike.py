"""ETL for Citi Bike mobility features.

Attempts to load from local zip; downloads from S3 if zip is a Git LFS pointer.
Falls back to placeholder on any failure so the pipeline keeps running.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.geospatial import lat_lon_to_nta

from .base import DatasetSpec, build_empty_frame

logger = logging.getLogger(__name__)

_RAW_ZIP = Path("data/raw/202603-citibike-tripdata.zip")
_S3_URL = "https://s3.amazonaws.com/tripdata/202603-citibike-tripdata.zip"
_TRIP_YEAR = 2026
_RAW_GLOB = "*-citibike-tripdata.zip"

DATASET_SPEC = DatasetSpec(
    name="citibike",
    owner="data",
    spatial_unit="nta",
    time_grain="year",
    description="Dock and trip activity as a walkability and lunch-demand proxy.",
    columns=("year", "nta_id", "trip_count", "station_count"),
)


def run_placeholder_etl() -> pd.DataFrame:
    return build_empty_frame(DATASET_SPEC)


def _transform(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate raw trip CSV to (nta_id, year) with trip_count and station_count."""
    lat_col = next((c for c in df.columns if "start_lat" in c.lower()), None)
    lon_col = next(
        (c for c in df.columns if "start_lng" in c.lower() or "start_lon" in c.lower()),
        None,
    )
    station_col = next(
        (
            c
            for c in df.columns
            if "start_station_id" in c.lower() or "start_station_name" in c.lower()
        ),
        None,
    )

    if lat_col is None or lon_col is None:
        logger.warning(
            "etl_citibike: lat/lon columns not found in %s", df.columns.tolist()
        )
        return build_empty_frame(DATASET_SPEC)

    df = df[[c for c in [lat_col, lon_col, station_col] if c]].copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    df["nta_id"] = lat_lon_to_nta(df[lat_col], df[lon_col])

    agg: dict[str, pd.Series] = {"trip_count": df.groupby("nta_id").size()}
    if station_col:
        agg["station_count"] = df.groupby("nta_id")[station_col].nunique()

    result = pd.DataFrame(agg).reset_index()
    if "station_count" not in result.columns:
        result["station_count"] = 0
    result["year"] = year
    return result[list(DATASET_SPEC.columns)]


def _load_zip(zip_bytes: bytes, year: int, nrows: int) -> pd.DataFrame:
    """Read the first CSV from a zip archive and transform it."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            logger.warning("etl_citibike: no CSV found inside zip")
            return build_empty_frame(DATASET_SPEC)
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, nrows=nrows)
    return _transform(df, year)


def _year_from_zip_name(path: Path) -> int | None:
    match = re.match(r"^(\d{4})\d{2}-citibike-tripdata\.zip$", path.name)
    if not match:
        return None
    return int(match.group(1))


def run_etl(limit: int = 50000) -> pd.DataFrame:
    """Load Citi Bike trip data.

    Prefers all local monthly zip snapshots under data/raw/*-citibike-tripdata.zip
    to maximize year coverage. Falls back to downloading the default snapshot.
    """
    local_zips = sorted(_RAW_ZIP.parent.glob(_RAW_GLOB))
    if local_zips:
        frames: list[pd.DataFrame] = []
        per_file_nrows = max(5000, limit // max(1, len(local_zips)))
        for zip_path in local_zips:
            with open(zip_path, "rb") as fh:
                magic = fh.read(2)
                if magic != b"PK":
                    logger.info("etl_citibike: skipping non-zip pointer %s", zip_path)
                    continue
                fh.seek(0)
                year = _year_from_zip_name(zip_path) or _TRIP_YEAR
                frame = _load_zip(fh.read(), year, per_file_nrows)
                if not frame.empty:
                    frames.append(frame)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged = (
                merged.groupby(["year", "nta_id"], as_index=False)
                .agg(trip_count=("trip_count", "sum"), station_count=("station_count", "max"))
                .sort_values(["year", "nta_id"])
                .reset_index(drop=True)
            )
            return merged[list(DATASET_SPEC.columns)]

    try:
        import requests

        logger.info("etl_citibike: downloading from %s", _S3_URL)
        resp = requests.get(_S3_URL, timeout=60)
        resp.raise_for_status()
        data = resp.content
        _RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
        _RAW_ZIP.write_bytes(data)
        return _load_zip(data, _TRIP_YEAR, limit)
    except Exception as exc:
        logger.warning(
            "etl_citibike: download failed (%s) — returning placeholder", exc
        )
        return run_placeholder_etl()
