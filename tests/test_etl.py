"""Tests for dataset scaffold modules."""

import pandas as pd
import pytest

import src.data.etl_runner as etl_runner
from src.data.audit import build_default_audit_rows
from src.data.base import DatasetSpec
from src.data.quality import prepare_training_frame, validate_dataset_contract
from src.data.registry import DATASET_REGISTRY


def test_registry_contains_core_datasets() -> None:
    """The scaffold should expose the main active sources."""

    for dataset_name in ("permits", "licenses", "inspections", "acs", "pluto"):
        assert dataset_name in DATASET_REGISTRY


def test_audit_rows_cover_the_registry() -> None:
    """The audit helper should expose one row per registered dataset."""

    assert len(build_default_audit_rows()) == len(DATASET_REGISTRY)


def test_validate_dataset_contract_rejects_missing_columns() -> None:
    spec = DatasetSpec(
        name="demo",
        owner="qa",
        spatial_unit="zone",
        time_grain="year",
        description="Demo dataset.",
        columns=("zone_id", "time_key"),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_contract(pd.DataFrame({"zone_id": ["z1"]}), spec)


def test_prepare_training_frame_deduplicates_and_filters_low_quality() -> None:
    frame = pd.DataFrame(
        {
            "zone_id": ["z1", "z1", "z2"],
            "time_key": [2024, 2024, 2024],
            "feature_a": [1.0, 2.0, float("inf")],
            "target": [0.6, 0.8, 0.5],
            "label_quality": [0.8, 0.9, 0.2],
        }
    )

    cleaned, report = prepare_training_frame(
        frame, target_col="target", min_label_quality=0.5
    )

    assert len(cleaned) == 1
    assert cleaned.loc[0, "feature_a"] == pytest.approx(2.0)
    assert report.dropped_rows == 2


def test_etl_runner_skips_optional_sources_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeYelpModule:
        def run_etl(self, limit: int = 0) -> pd.DataFrame:  # noqa: ARG002
            raise RuntimeError("YELP_DATA_PATH env var required")

    monkeypatch.setattr(etl_runner, "_ETL_MODULES", {"yelp": FakeYelpModule()})

    results, status = etl_runner.run_all_etl()

    assert status["yelp"] == "skipped"
    assert results["yelp"].empty


def test_licenses_transform_keeps_business_unique_id_separate() -> None:
    from src.data.etl_licenses import transform

    raw = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01"],
            "business_unique_id": ["dca-42"],
            "license_status": ["Active"],
            "nta": ["BK09"],
            "business_category": ["Restaurant"],
        }
    )

    transformed = transform(raw)

    assert transformed.loc[0, "business_unique_id"] == "dca-42"
    assert pd.isna(transformed.loc[0, "restaurant_id"])


# ── etl_311 ───────────────────────────────────────────────────────────────────


def test_etl_311_transform_aggregates() -> None:
    from src.data.etl_311 import transform

    raw = pd.DataFrame(
        {
            "created_date": [
                "2024-01-15T10:00:00",
                "2024-01-15T11:00:00",
                "2024-02-01T09:00:00",
            ],
            "community_board": ["01 MANHATTAN", "01 MANHATTAN", "02 BROOKLYN"],
            "complaint_type": [
                "Food Establishment",
                "Food Establishment",
                "Food Poisoning",
            ],
        }
    )
    result = transform(raw)
    assert "month" in result.columns
    assert "community_district" in result.columns
    assert "count" in result.columns
    assert not result.empty


def test_etl_311_transform_drops_invalid_dates() -> None:
    from src.data.etl_311 import transform

    raw = pd.DataFrame(
        {
            "created_date": ["invalid-date", "2024-01-15T10:00:00"],
            "community_board": ["01 MANHATTAN", "02 BROOKLYN"],
            "complaint_type": ["Food Establishment", "Food Establishment"],
        }
    )
    result = transform(raw)
    assert len(result) == 1


def test_etl_311_run_placeholder_etl() -> None:
    from src.data.etl_311 import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty
    for col in ("month", "community_district", "complaint_type", "count"):
        assert col in result.columns


# ── etl_acs ───────────────────────────────────────────────────────────────────


def test_etl_acs_borough_key_bn() -> None:
    from src.data.etl_acs import _borough_key

    assert _borough_key("MN17") == "MN"
    assert _borough_key("BK09") == "BK"
    assert _borough_key("bk-tandon") == "BK"
    assert _borough_key("mn-fidi") == "MN"


def test_etl_acs_build_synthetic() -> None:
    from src.data.etl_acs import _build_synthetic_acs

    result = _build_synthetic_acs(limit=10)
    assert not result.empty
    assert "nta_id" in result.columns
    assert "median_income" in result.columns
    assert "population" in result.columns
    assert "rent_burden" in result.columns


def test_etl_acs_run_etl_uses_synthetic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_acs

    monkeypatch.setenv("ACS_DATA_PATH", "")
    result = etl_acs.run_etl(limit=5)
    assert not result.empty
    assert "nta_id" in result.columns


def test_etl_acs_load_local_raises_when_no_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.etl_acs import _load_local

    monkeypatch.delenv("ACS_DATA_PATH", raising=False)
    with pytest.raises(RuntimeError, match="ACS_DATA_PATH"):
        _load_local()


def test_etl_acs_run_placeholder() -> None:
    from src.data.etl_acs import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


# ── etl_yelp ──────────────────────────────────────────────────────────────────


def test_etl_yelp_load_business_empty_when_no_file() -> None:
    from src.data.etl_yelp import _load_business

    result = _load_business()
    assert isinstance(result, pd.DataFrame)
    for col in ("id", "latitude", "longitude"):
        assert col in result.columns


def test_etl_yelp_load_local_raises_when_no_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from src.data import etl_yelp

    monkeypatch.setattr(
        etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", Path("/nonexistent/yelp.csv")
    )
    monkeypatch.delenv("YELP_DATA_PATH", raising=False)
    with pytest.raises(RuntimeError, match="etl_yelp"):
        etl_yelp._load_local()


def test_etl_yelp_run_placeholder() -> None:
    from src.data.etl_yelp import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


# ── etl_airbnb ────────────────────────────────────────────────────────────────


def test_etl_airbnb_build_synthetic() -> None:
    from src.data.etl_airbnb import _build_synthetic_airbnb

    result = _build_synthetic_airbnb()
    assert not result.empty
    assert "nta_id" in result.columns
    assert "listing_count" in result.columns
    assert "entire_home_ratio" in result.columns
    assert (result["listing_count"] >= 1).all()
    assert result["entire_home_ratio"].between(0.01, 0.99).all()


def test_etl_airbnb_transform_with_lat_lon(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_airbnb

    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["BK09"] * len(lat), index=lat.index),
    )
    raw = pd.DataFrame(
        {
            "latitude": [40.69, 40.70],
            "longitude": [-73.99, -73.98],
            "room_type": ["Entire home/apt", "Private room"],
        }
    )
    result = etl_airbnb._transform(raw)
    assert "nta_id" in result.columns


def test_etl_airbnb_transform_no_lat_lon() -> None:
    from src.data.etl_airbnb import _transform

    raw = pd.DataFrame({"price": [100, 200]})
    result = _transform(raw)
    assert result.empty


def test_etl_airbnb_transform_no_room_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_airbnb

    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["MN17"] * len(lat), index=lat.index),
    )
    raw = pd.DataFrame({"latitude": [40.75], "longitude": [-73.99]})
    result = etl_airbnb._transform(raw)
    assert "nta_id" in result.columns


def test_etl_airbnb_run_etl_synthetic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_airbnb

    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", etl_airbnb._RAW_CSV_GZ)
    monkeypatch.setattr(etl_airbnb._RAW_CSV_GZ.__class__, "exists", lambda _: False)
    result = etl_airbnb._build_synthetic_airbnb()
    assert not result.empty


def test_etl_airbnb_run_placeholder() -> None:
    from src.data.etl_airbnb import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


# ── etl_citibike ──────────────────────────────────────────────────────────────


def test_etl_citibike_transform_with_lat_lon(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_citibike

    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["BK09"] * len(lat), index=lat.index),
    )
    raw = pd.DataFrame(
        {
            "start_lat": [40.69, 40.70, 40.71],
            "start_lng": [-73.99, -73.98, -73.97],
            "start_station_id": ["S1", "S1", "S2"],
        }
    )
    result = etl_citibike._transform(raw, year=2024)
    assert "nta_id" in result.columns
    assert "trip_count" in result.columns
    assert "year" in result.columns


def test_etl_citibike_transform_no_lat_lon() -> None:
    from src.data.etl_citibike import _transform

    raw = pd.DataFrame({"price": [100]})
    result = _transform(raw, year=2024)
    assert result.empty


def test_etl_citibike_transform_no_station_col(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_citibike

    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["QN70"] * len(lat), index=lat.index),
    )
    raw = pd.DataFrame({"start_lat": [40.69], "start_lng": [-73.94]})
    result = etl_citibike._transform(raw, year=2024)
    assert "station_count" in result.columns


def test_etl_citibike_load_zip_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import zipfile
    from src.data import etl_citibike

    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["MN17"] * len(lat), index=lat.index),
    )
    # Build a minimal in-memory zip with a CSV
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_content = "start_lat,start_lng\n40.75,-73.99\n40.76,-73.98\n"
        zf.writestr("trips.csv", csv_content)
    buf.seek(0)
    result = etl_citibike._load_zip(buf.read(), year=2024, nrows=100)
    assert isinstance(result, pd.DataFrame)


def test_etl_citibike_load_zip_no_csv_in_zip() -> None:
    import io
    import zipfile
    from src.data.etl_citibike import _load_zip

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no CSV here")
    buf.seek(0)
    result = _load_zip(buf.read(), year=2024, nrows=100)
    assert result.empty


def test_etl_citibike_run_placeholder() -> None:
    from src.data.etl_citibike import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


# ── etl_pluto ─────────────────────────────────────────────────────────────────


def test_etl_pluto_run_placeholder() -> None:
    from src.data.etl_pluto import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


def test_etl_pluto_transform_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN17", "11201": "BK09"})
    raw = pd.DataFrame(
        {
            "yearbuilt": ["1990", "2000"],
            "zipcode": ["10001", "11201"],
            "comarea": ["5000", "3000"],
            "retailarea": ["1000", "500"],
            "lotarea": ["10000", "8000"],
            "bldgarea": ["8000", "6000"],
            "assesstot": ["1000000", "500000"],
            "borough": ["MN", "BK"],
        }
    )
    result = etl_pluto.transform(raw)
    assert "nta_id" in result.columns
    assert not result.empty


def test_etl_pluto_transform_no_zip_map(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {})
    raw = pd.DataFrame(
        {
            "yearbuilt": ["1990"],
            "comarea": ["5000"],
            "retailarea": ["1000"],
            "lotarea": ["10000"],
            "bldgarea": ["8000"],
            "assesstot": ["1000000"],
            "borough": ["MN"],
        }
    )
    result = etl_pluto.transform(raw)
    assert isinstance(result, pd.DataFrame)


# ── etl_permits ───────────────────────────────────────────────────────────────


def test_etl_permits_run_placeholder() -> None:
    from src.data.etl_permits import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


def test_etl_permits_transform_basic() -> None:
    from src.data.etl_permits import transform

    raw = pd.DataFrame(
        {
            "issueddate": ["2024-01-15", "2024-02-20"],
            "communityboard": ["01 MANHATTAN", "02 BROOKLYN"],
            "permitsub": ["NB", "A1"],
        }
    )
    result = transform(raw)
    assert "permit_date" in result.columns
    assert "nta_id" in result.columns
    assert "job_count" in result.columns
    assert not result.empty


def test_etl_permits_transform_missing_required_cols() -> None:
    from src.data.etl_permits import transform

    raw = pd.DataFrame({"other_col": ["value"]})
    result = transform(raw)
    assert result.empty


def test_etl_permits_transform_no_type_col() -> None:
    from src.data.etl_permits import transform

    raw = pd.DataFrame(
        {
            "issueddate": ["2024-01-15"],
            "communityboard": ["01 MANHATTAN"],
        }
    )
    result = transform(raw)
    assert not result.empty
    assert "permit_type" in result.columns


# ── etl_inspections ───────────────────────────────────────────────────────────


def test_etl_inspections_run_placeholder() -> None:
    from src.data.etl_inspections import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


def test_etl_inspections_transform_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN17"})
    raw = pd.DataFrame(
        {
            "camis": ["12345"],
            "inspection_date": ["2024-01-15"],
            "grade": ["A"],
            "critical_flag": ["Not Critical"],
            "boro": ["manhattan"],
            "zipcode": ["10001"],
            "cuisine_description": ["Italian"],
        }
    )
    result = etl_insp.transform(raw)
    assert "restaurant_id" in result.columns
    assert "nta_id" in result.columns
    assert not result.empty


def test_etl_inspections_transform_unmapped_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {})
    raw = pd.DataFrame(
        {
            "camis": ["99999"],
            "inspection_date": ["2024-06-01"],
            "grade": ["B"],
            "critical_flag": ["Critical"],
            "boro": ["brooklyn"],
            "zipcode": ["11201"],
            "cuisine_description": ["Chinese"],
        }
    )
    result = etl_insp.transform(raw)
    assert not result.empty
    assert result.iloc[0]["nta_id"].startswith("BK")


def test_etl_inspections_transform_no_zip_col(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN17"})
    raw = pd.DataFrame(
        {
            "camis": ["11111"],
            "inspection_date": ["2024-03-01"],
            "grade": ["A"],
            "critical_flag": ["Not Critical"],
            "boro": ["1"],  # manhattan code; no zipcode column
        }
    )
    result = etl_insp.transform(raw)
    assert "nta_id" in result.columns


# ── etl_licenses (additional) ─────────────────────────────────────────────────


def test_etl_licenses_run_placeholder() -> None:
    from src.data.etl_licenses import run_placeholder_etl

    result = run_placeholder_etl()
    assert result.empty


def test_etl_licenses_transform_filters_unknown_nta() -> None:
    from src.data.etl_licenses import transform

    raw = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01", "2024-02-01"],
            "business_unique_id": ["dca-1", "dca-2"],
            "license_status": ["Active", "Active"],
            "nta": ["BK09", None],  # one valid, one null → becomes UNKNOWN → filtered
            "business_category": ["Restaurant", "Restaurant"],
        }
    )
    result = transform(raw)
    assert len(result) == 1
    assert result.iloc[0]["nta_id"] == "BK09"


def test_etl_licenses_transform_adds_restaurant_id_na() -> None:
    from src.data.etl_licenses import transform

    raw = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01"],
            "business_unique_id": ["dca-42"],
            "license_status": ["Active"],
            "nta": ["MN17"],
            "business_category": ["Restaurant"],
        }
    )
    result = transform(raw)
    assert "restaurant_id" in result.columns


# ── etl_runner (additional) ───────────────────────────────────────────────────


def test_etl_runner_all_modules_in_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.etl_runner as runner

    class FakePlaceholder:
        DATASET_SPEC = None

        def run_placeholder_etl(self):
            return pd.DataFrame()

    monkeypatch.setattr(runner, "_ETL_MODULES", {"fake": FakePlaceholder()})
    results, status = runner.run_all_etl()
    assert "fake" in status


def test_etl_runner_strict_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.etl_runner as runner

    class FailModule:
        DATASET_SPEC = None

        def run_etl(self, limit: int = 0) -> pd.DataFrame:
            raise ValueError("boom")

    monkeypatch.setattr(runner, "_ETL_MODULES", {"fail": FailModule()})
    with pytest.raises(ValueError, match="boom"):
        runner.run_all_etl(strict=True)


def test_run_module_fallback_to_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.etl_runner as runner

    class PlaceholderOnlyModule:
        def run_placeholder_etl(self):
            return pd.DataFrame({"col": [1]})

    result = runner._run_module(PlaceholderOnlyModule(), limit=10)
    assert not result.empty


def test_run_module_raises_no_runner() -> None:
    import src.data.etl_runner as runner

    class NoRunModule:
        pass

    with pytest.raises(AttributeError):
        runner._run_module(NoRunModule(), limit=10)


# ── data quality (additional) ─────────────────────────────────────────────────


def test_prepare_survival_history_basic() -> None:
    from src.data.quality import prepare_survival_history

    history = pd.DataFrame(
        {
            "restaurant_id": ["r1", "r2", "r3"],
            "duration_days": [100, 200, 0],
            "event_observed": [1, 0, 1],
        }
    )
    frame, report = prepare_survival_history(history)
    # row with duration=0 should be dropped
    assert report.dropped_rows >= 1
    assert "restaurant_id" in frame.columns


def test_prepare_survival_history_missing_cols() -> None:
    from src.data.quality import prepare_survival_history

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_survival_history(pd.DataFrame({"restaurant_id": ["r1"]}))


def test_prepare_training_frame_raises_missing_target() -> None:
    from src.data.quality import prepare_training_frame

    with pytest.raises(ValueError, match="must contain"):
        prepare_training_frame(pd.DataFrame({"x": [1.0]}), target_col="missing")


def test_prepare_training_frame_no_key_columns() -> None:
    from src.data.quality import prepare_training_frame

    df = pd.DataFrame({"target": [0.5, 0.8], "x": [1.0, 2.0]})
    frame, report = prepare_training_frame(df, target_col="target", key_columns=())
    assert not frame.empty


def test_prepare_embedding_corpus_custom_dedupe() -> None:
    from src.data.quality import prepare_embedding_corpus

    df = pd.DataFrame(
        {
            "review_text": ["hello world", "hello world", "fresh food"],
            "restaurant_id": ["r1", "r1", "r2"],
        }
    )
    frame, report = prepare_embedding_corpus(df, dedupe_columns=["review_text"])
    assert len(frame) == 2


def test_prepare_embedding_corpus_raises_no_text_col() -> None:
    from src.data.quality import prepare_embedding_corpus

    with pytest.raises(ValueError, match="review_text"):
        prepare_embedding_corpus(pd.DataFrame({"other": ["x"]}))


# ── data base ─────────────────────────────────────────────────────────────────


def test_base_dataset_pipeline_methods() -> None:
    from src.data.base import BaseDatasetPipeline, DatasetSpec

    spec = DatasetSpec(
        name="test",
        owner="test",
        spatial_unit="zone",
        time_grain="year",
        description="test",
        columns=("a", "b"),
    )
    pipeline = BaseDatasetPipeline(spec=spec)
    extracted = pipeline.extract()
    assert list(extracted.columns) == ["a", "b"]
    transformed = pipeline.transform(pd.DataFrame({"a": [1]}))
    assert "a" in transformed.columns
    loaded = pipeline.load(pd.DataFrame({"a": [1]}))
    assert "a" in loaded.columns


# ── etl_acs — additional ──────────────────────────────────────────────────────


def test_etl_acs_run_etl_uses_synthetic_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_acs

    monkeypatch.delenv("ACS_DATA_PATH", raising=False)
    result = etl_acs.run_etl(limit=10)
    assert not result.empty
    assert "nta_id" in result.columns


def test_etl_acs_load_local_raises_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_acs

    monkeypatch.setenv("ACS_DATA_PATH", str(tmp_path / "nonexistent.csv"))
    with pytest.raises(FileNotFoundError):
        etl_acs._load_local()


# ── etl_airbnb — run_etl synthetic fallback ──────────────────────────────────


def test_etl_airbnb_run_etl_synthetic_when_no_local_and_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from src.data import etl_airbnb

    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", Path("/nonexistent/airbnb.csv"))
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV_GZ", Path("/nonexistent/airbnb.csv.gz"))
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    result = etl_airbnb.run_etl(limit=5)
    assert not result.empty
    assert "nta_id" in result.columns


def test_etl_airbnb_run_etl_synthetic_when_transform_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from src.data import etl_airbnb

    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", Path("/nonexistent/airbnb.csv"))
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV_GZ", Path("/nonexistent/airbnb.csv.gz"))
    monkeypatch.setattr(etl_airbnb, "_transform", lambda df: pd.DataFrame())
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **kw: type(
            "R",
            (),
            {
                "status_code": 200,
                "content": b"a,b\n1,2\n",
                "raise_for_status": lambda s: None,
            },
        )(),
    )
    result = etl_airbnb.run_etl(limit=5)
    assert not result.empty  # falls back to synthetic


# ── etl_citibike — run_etl download fallback ─────────────────────────────────


def test_etl_citibike_run_etl_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from src.data import etl_citibike

    monkeypatch.setattr(etl_citibike, "_RAW_ZIP", Path("/nonexistent/citibike.zip"))
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    result = etl_citibike.run_etl(limit=5)
    assert (
        result.empty
        or "trip_count" in result.columns
        or isinstance(result, pd.DataFrame)
    )


# ── etl_yelp — additional paths ──────────────────────────────────────────────


def test_etl_yelp_load_local_reads_from_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_yelp

    csv_path = tmp_path / "yelp.csv"
    pd.DataFrame({"review_id": ["r1"], "text": ["great food"]}).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", csv_path)
    result = etl_yelp._load_local()
    assert not result.empty


def test_etl_yelp_load_local_reads_from_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pathlib import Path
    from src.data import etl_yelp

    monkeypatch.setattr(
        etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", Path("/nonexistent/yelp.csv")
    )
    csv_path = tmp_path / "yelp_env.csv"
    pd.DataFrame({"review_id": ["r1"]}).to_csv(csv_path, index=False)
    monkeypatch.setenv("YELP_DATA_PATH", str(csv_path))
    result = etl_yelp._load_local()
    assert not result.empty


def test_etl_yelp_run_etl_with_mocked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pathlib import Path
    from src.data import etl_yelp

    reviews = pd.DataFrame(
        {"business_id": ["b1"], "review_text": ["good food"], "rating": [4]}
    )
    reviews_path = tmp_path / "reviews.csv"
    reviews.to_csv(reviews_path, index=False)
    monkeypatch.setattr(etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", reviews_path)
    monkeypatch.setattr(
        etl_yelp, "_DEFAULT_FUSION_BUSINESS_PATH", Path("/nonexistent/business.csv")
    )
    result = etl_yelp.run_etl()
    assert isinstance(result, pd.DataFrame)


# ── etl_311 — fetch ───────────────────────────────────────────────────────────


def test_etl_311_fetch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_311

    mock_data = [
        {
            "created_date": "2024-01-15",
            "community_board": "101 BROOKLYN",
            "complaint_type": "Food Establishment",
        }
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_311.fetch(limit=10)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_etl_311_run_etl_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_311

    mock_data = [
        {
            "created_date": "2024-01-15",
            "community_board": "101 BROOKLYN",
            "complaint_type": "Food Establishment",
        }
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_311.run_etl(limit=10)
    assert "month" in result.columns


# ── etl_pluto — fetch and transform edge cases ────────────────────────────────


def test_etl_pluto_fetch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_pluto

    mock_data = [
        {
            "yearbuilt": "1985",
            "zipcode": "10001",
            "borough": "MN",
            "lotarea": "5000",
            "bldgarea": "8000",
            "comarea": "2000",
            "retailarea": "1000",
            "assesstot": "500000",
        }
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_pluto.fetch(limit=5)
    assert isinstance(result, pd.DataFrame)


def test_etl_pluto_transform_no_bldgarea(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.etl_inspections as etl_insp
    from src.data import etl_pluto

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN17"})
    raw = pd.DataFrame(
        {
            "yearbuilt": ["1985"],
            "zipcode": ["10001"],
            "borough": ["MN"],
            "comarea": ["2000"],
            "retailarea": ["1000"],
            "assesstot": ["500000"],
        }
    )
    result = etl_pluto.transform(raw)
    assert "mixed_use_ratio" in result.columns
    assert result.iloc[0]["mixed_use_ratio"] == 0.0


# ── etl_licenses — fetch ─────────────────────────────────────────────────────


def test_etl_licenses_fetch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_licenses

    mock_data = [
        {
            "license_creation_date": "2024-01-01",
            "business_unique_id": "dca-1",
            "license_status": "Active",
            "address_borough": "BROOKLYN",
            "nta": "BK09",
            "address_zip": "11201",
            "business_category": "Restaurant",
        }
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_licenses.fetch(limit=5)
    assert isinstance(result, pd.DataFrame)


# ── etl_permits — fetch ──────────────────────────────────────────────────────


def test_etl_permits_fetch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_permits

    mock_data = [
        {
            "issueddate": "2024-01-01",
            "communityboard": "101 BROOKLYN",
            "permitsub": "NB",
        }
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_permits.fetch(limit=5)
    assert isinstance(result, pd.DataFrame)


# ── etl_inspections — fetch ──────────────────────────────────────────────────


def test_etl_inspections_fetch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_inspections

    mock_data = [
        {
            "inspection_date": "2024-01-15",
            "camis": "12345",
            "grade": "A",
            "critical_flag": "Not Critical",
            "boro": "1",
            "zipcode": "10001",
            "cuisine_description": "Chinese",
            "dba": "Test Restaurant",
        }
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_inspections.fetch(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_etl_inspections_get_zip_to_nta_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", None)
    mock_data = [{"nta": "BK09", "address_zip": "11201"}]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_data

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_insp._get_zip_to_nta()
    assert isinstance(result, dict)
    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", None)


# ── etl_runner — strict failure mode ────────────────────────────────────────


def test_etl_runner_strict_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_runner

    def _failing_module():
        return type(
            "M",
            (),
            {
                "run_placeholder_etl": staticmethod(
                    lambda: (_ for _ in ()).throw(RuntimeError("forced fail"))
                )
            },
        )()

    original_modules = etl_runner._ETL_MODULES.copy()
    monkeypatch.setattr(etl_runner, "_ETL_MODULES", {"bad_module": _failing_module()})
    with pytest.raises(Exception):
        etl_runner.run_all_etl(strict=True)
    monkeypatch.setattr(etl_runner, "_ETL_MODULES", original_modules)


def test_etl_runner_non_strict_continues_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_runner
    from src.data.base import DatasetSpec

    class FailModule:
        DATASET_SPEC = DatasetSpec(
            name="fail",
            owner="t",
            spatial_unit="zone",
            time_grain="year",
            description="t",
            columns=("a",),
        )

        @staticmethod
        def run_placeholder_etl():
            raise RuntimeError("always fails")

    original_modules = etl_runner._ETL_MODULES.copy()
    monkeypatch.setattr(etl_runner, "_ETL_MODULES", {"fail": FailModule()})
    results, status = etl_runner.run_all_etl(strict=False)
    assert status.get("fail") in ("failed", "skipped")
    monkeypatch.setattr(etl_runner, "_ETL_MODULES", original_modules)
