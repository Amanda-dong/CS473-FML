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
    assert transformed.loc[0, "restaurant_id"] == "dca-42"


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

    assert _borough_key("MN0604") == "MN"
    assert _borough_key("BK09") == "BK"
    assert _borough_key("bk-tandon") == "BK"
    assert _borough_key("mn-fidi") == "MN"


def test_etl_acs_run_etl_falls_back_when_no_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_acs

    monkeypatch.setenv("ACS_DATA_PATH", "")
    # run_etl() now gracefully falls back to canonical CSV or placeholder
    result = etl_acs.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)


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
        lambda lat, lon: pd.Series(["MN0604"] * len(lat), index=lat.index),
    )
    raw = pd.DataFrame({"latitude": [40.75], "longitude": [-73.99]})
    result = etl_airbnb._transform(raw)
    assert "nta_id" in result.columns


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
        lambda lat, lon: pd.Series(["MN0604"] * len(lat), index=lat.index),
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

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN0604", "11201": "BK09"})
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

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN0604"})
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

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN0604"})
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


def test_etl_licenses_transform_restaurant_id_from_business_unique_id() -> None:
    from src.data.etl_licenses import transform

    raw = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01"],
            "business_unique_id": ["dca-42"],
            "license_status": ["Active"],
            "nta": ["MN0604"],
            "business_category": ["Restaurant"],
        }
    )
    result = transform(raw)
    assert "restaurant_id" in result.columns
    assert result.iloc[0]["restaurant_id"] == "dca-42"


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


def test_etl_acs_run_etl_falls_back_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_acs

    monkeypatch.delenv("ACS_DATA_PATH", raising=False)
    # run_etl() falls back to canonical CSV or placeholder when env var absent
    result = etl_acs.run_etl(limit=10)
    assert isinstance(result, pd.DataFrame)


def test_etl_acs_load_local_raises_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_acs

    monkeypatch.setenv("ACS_DATA_PATH", str(tmp_path / "nonexistent.csv"))
    with pytest.raises(FileNotFoundError):
        etl_acs._load_local()


# ── etl_airbnb — run_etl synthetic fallback ──────────────────────────────────


def test_etl_airbnb_run_etl_raises_when_download_fails(
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
    with pytest.raises(ConnectionError, match="offline"):
        etl_airbnb.run_etl(limit=5)


def test_etl_airbnb_run_etl_raises_when_transform_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import gzip
    from pathlib import Path

    from src.data import etl_airbnb

    dl_path = tmp_path / "airbnb.csv.gz"
    gz_content = gzip.compress(b"a,b\n1,2\n")
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", Path("/nonexistent/airbnb.csv"))
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV_GZ", dl_path)
    monkeypatch.setattr(etl_airbnb, "_transform", lambda df: pd.DataFrame())
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **kw: type(
            "R",
            (),
            {
                "status_code": 200,
                "content": gz_content,
                "raise_for_status": lambda s: None,
            },
        )(),
    )
    with pytest.raises(RuntimeError, match="empty frame"):
        etl_airbnb.run_etl(limit=5)


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

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN0604"})
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


# ── etl_runner — deprecated, missing-spec, ok, and generic-exception paths ────


def _make_etl_module(
    name: str,
    *,
    df: "pd.DataFrame | None" = None,
    raises: "Exception | None" = None,
    status: str = "planned",
    drop_spec: bool = False,
) -> object:
    spec = (
        None
        if drop_spec
        else DatasetSpec(
            name=name,
            owner="t",
            spatial_unit="zone",
            time_grain="year",
            description="t",
            columns=("a",),
            status=status,
        )
    )
    _ret = df if df is not None else pd.DataFrame({"a": [1]})
    _raises = raises

    def _run(limit: int = 0) -> pd.DataFrame:
        if _raises is not None:
            raise _raises
        return _ret

    klass = type(name, (), {"run_etl": staticmethod(_run)})
    if spec is not None:
        klass.DATASET_SPEC = spec  # type: ignore[attr-defined]
    return klass()


def test_etl_runner_skips_deprecated_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_runner

    monkeypatch.setattr(
        etl_runner,
        "_ETL_MODULES",
        {
            "old": _make_etl_module("old", status="deprecated"),
        },
    )
    _, status = etl_runner.run_all_etl()
    assert status["old"] == "skipped"


def test_etl_runner_ok_path_with_valid_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_runner

    monkeypatch.setattr(
        etl_runner,
        "_ETL_MODULES",
        {
            "ok_mod": _make_etl_module("ok_mod", df=pd.DataFrame({"a": [1, 2, 3]})),
        },
    )
    results, status = etl_runner.run_all_etl()
    assert status["ok_mod"] == "ok"
    assert len(results["ok_mod"]) == 3


def test_etl_runner_handles_missing_spec_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_runner

    monkeypatch.setattr(
        etl_runner,
        "_ETL_MODULES",
        {
            "no_spec": _make_etl_module("no_spec", drop_spec=True),
        },
    )
    _, status = etl_runner.run_all_etl(strict=False)
    assert status["no_spec"] == "failed"


def test_etl_runner_generic_exception_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_runner

    monkeypatch.setattr(
        etl_runner,
        "_ETL_MODULES",
        {
            "boom": _make_etl_module("boom", raises=ValueError("generic error")),
        },
    )
    with pytest.raises(ValueError, match="generic error"):
        etl_runner.run_all_etl(strict=True)


def test_etl_runner_empty_module_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_runner

    monkeypatch.setattr(
        etl_runner,
        "_ETL_MODULES",
        {
            "emp": _make_etl_module("emp", df=pd.DataFrame({"a": []})),
        },
    )
    _, status = etl_runner.run_all_etl()
    assert status["emp"] == "empty"


# ── etl_acs — borough key fallback and local load ─────────────────────────────


def test_etl_acs_borough_key_unknown_prefix_returns_mn() -> None:
    from src.data.etl_acs import _borough_key

    assert _borough_key("XX99") == "MN"


def test_etl_acs_load_local_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from src.data import etl_acs

    csv_path = tmp_path / "acs.csv"
    pd.DataFrame({"nta_id": ["BK09"], "median_income": [60000]}).to_csv(
        csv_path, index=False
    )
    monkeypatch.setenv("ACS_DATA_PATH", str(csv_path))
    result = etl_acs._load_local()
    assert not result.empty


def test_etl_acs_run_etl_falls_back_when_local_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_acs

    monkeypatch.setattr(etl_acs, "_load_local", lambda: pd.DataFrame())
    # run_etl() catches the empty-frame error and falls back to
    # canonical CSV or placeholder
    result = etl_acs.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)


def test_etl_acs_run_etl_success_from_local(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_acs

    fake = pd.DataFrame(
        {
            "year": [2024],
            "nta_id": ["BK09"],
            "median_income": [60000.0],
            "population": [5000.0],
            "rent_burden": [0.35],
        }
    )
    monkeypatch.setattr(etl_acs, "_load_local", lambda: fake)
    result = etl_acs.run_etl(limit=10)
    assert len(result) == 1


# ── etl_airbnb — _read_local success and download path ───────────────────────


def test_etl_airbnb_read_local_existing_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pathlib import Path
    from src.data import etl_airbnb

    good = tmp_path / "listings.csv"
    pd.DataFrame({"id": ["a1"], "neighbourhood_cleansed": ["Bushwick"]}).to_csv(
        good, index=False
    )
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", good)
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV_GZ", Path("/nonexistent.gz"))
    result = etl_airbnb._read_local(limit=10)
    assert result is not None


def test_etl_airbnb_run_etl_raises_when_transform_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import requests
    from pathlib import Path

    from src.data import etl_airbnb

    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", Path("/nonexistent/listings.csv"))
    dl_path = tmp_path / "listings.csv"
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV_GZ", dl_path)

    class MockResponse:
        content = b"col1,col2\n1,2\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    monkeypatch.setattr(etl_airbnb, "_transform", lambda df: pd.DataFrame())
    with pytest.raises(RuntimeError, match="empty frame"):
        etl_airbnb.run_etl(limit=5)


# ── etl_citibike — PK zip path and download mock ─────────────────────────────


def test_etl_citibike_run_etl_with_real_zip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import io
    import zipfile
    from src.data import etl_citibike

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("trips.csv", "col1,col2\n1,2\n")
    zip_path = tmp_path / "202603-citibike-tripdata.zip"
    zip_path.write_bytes(buf.getvalue())
    monkeypatch.setattr(etl_citibike, "_RAW_ZIP", zip_path)
    result = etl_citibike.run_etl(limit=10)
    assert isinstance(result, pd.DataFrame)


def test_etl_citibike_run_etl_download_mock(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import io
    import requests
    import zipfile
    from src.data import etl_citibike

    dl_zip = tmp_path / "citibike.zip"
    monkeypatch.setattr(etl_citibike, "_RAW_ZIP", dl_zip)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("trips.csv", "col1,col2\n1,2\n")

    class MockResponse:
        content = buf.getvalue()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    result = etl_citibike.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)


# ── etl_inspections — _get_zip_to_nta exception path ─────────────────────────


def test_etl_inspections_zip_to_nta_exception_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", None)
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network error")),
    )
    result = etl_insp._get_zip_to_nta()
    assert result == {}
    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", None)


# ── etl_yelp — JSON load, business exception, and business merge ──────────────


def test_etl_yelp_nonexistent_data_path_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_yelp

    monkeypatch.setenv("YELP_DATA_PATH", "/nonexistent_path_xyz/reviews.json")
    monkeypatch.setattr(
        etl_yelp,
        "_DEFAULT_FUSION_REVIEW_PATH",
        __import__("pathlib").Path("/nonexistent"),
    )
    with pytest.raises(FileNotFoundError):
        etl_yelp._load_local()


def test_etl_yelp_load_local_json_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_yelp

    json_path = tmp_path / "reviews.json"
    pd.DataFrame({"review_id": ["r1"], "text": ["good"]}).to_json(
        json_path, orient="records", lines=True
    )
    monkeypatch.setattr(etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", json_path)
    result = etl_yelp._load_local()
    assert isinstance(result, pd.DataFrame)


def test_etl_yelp_load_business_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_yelp

    bad = tmp_path / "business.csv"
    bad.write_text("wrong_col1,wrong_col2\n1,2\n")
    monkeypatch.setattr(etl_yelp, "_DEFAULT_FUSION_BUSINESS_PATH", bad)
    result = etl_yelp._load_business()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["id", "latitude", "longitude"]


def test_etl_yelp_run_etl_merges_business(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_yelp

    reviews = pd.DataFrame(
        {"business_id": ["b1"], "review_text": ["salad"], "rating": [4]}
    )
    reviews_path = tmp_path / "reviews.csv"
    reviews.to_csv(reviews_path, index=False)
    monkeypatch.setattr(etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", reviews_path)

    business = pd.DataFrame({"id": ["b1"], "latitude": [40.7], "longitude": [-73.9]})
    biz_path = tmp_path / "business.csv"
    business.to_csv(biz_path, index=False)
    monkeypatch.setattr(etl_yelp, "_DEFAULT_FUSION_BUSINESS_PATH", biz_path)

    result = etl_yelp.run_etl()
    assert "latitude" in result.columns


# ── quality — embedding corpus dedupe fallback and non-numeric training ────────


def test_prepare_embedding_corpus_no_standard_cols_dedupe_fallback() -> None:
    from src.data.quality import prepare_embedding_corpus

    df = pd.DataFrame({"review_text": ["hello world", "another review"]})
    result, report = prepare_embedding_corpus(df)
    assert not result.empty


def test_prepare_embedding_corpus_explicit_dedupe_missing_cols() -> None:
    from src.data.quality import prepare_embedding_corpus

    df = pd.DataFrame({"review_text": ["hello world", "another review"]})
    result, report = prepare_embedding_corpus(df, dedupe_columns=["nonexistent_col"])
    assert not result.empty


def test_prepare_training_frame_raises_on_non_numeric_column() -> None:
    from src.data.quality import prepare_training_frame

    frame = pd.DataFrame(
        {
            "zone_id": ["z1"],
            "time_key": [2024],
            "target": [0.8],
            "label_quality": [1.0],
            "cat_feature": ["some_string"],
        }
    )
    with pytest.raises(ValueError, match="non-numeric"):
        prepare_training_frame(frame, target_col="target")


# ── etl_yelp — utility functions ─────────────────────────────────────────────


def test_etl_yelp_build_headers() -> None:
    from src.data.etl_yelp import _build_headers

    headers = _build_headers("my-api-key")
    assert headers["Authorization"] == "Bearer my-api-key"


@pytest.mark.parametrize(
    "lat,lon,term,offset",
    [
        (40.7, -73.9, "healthy food", 0),
        (40.8, -74.0, "halal", 50),
    ],
)
def test_etl_yelp_build_params(lat: float, lon: float, term: str, offset: int) -> None:
    from src.data.etl_yelp import _build_params

    anchor = {"latitude": lat, "longitude": lon, "anchor_name": "test"}
    params = _build_params(anchor, term, offset)
    assert params["term"] == term
    assert params["latitude"] == lat
    assert params["longitude"] == lon
    assert params["offset"] == offset
    assert params["categories"] == "restaurants"


def test_etl_yelp_extract_businesses_basic() -> None:
    from src.data.etl_yelp import _extract_businesses

    businesses = [
        {
            "id": "abc123",
            "name": "Test Restaurant",
            "categories": [{"title": "Salads"}, {"title": "Healthy"}],
            "rating": 4.5,
            "review_count": 100,
            "price": "$$",
            "is_closed": False,
            "coordinates": {"latitude": 40.7, "longitude": -73.9},
        }
    ]
    rows = _extract_businesses(businesses, "healthy", "downtown")
    assert len(rows) == 1
    assert rows[0]["id"] == "abc123"
    assert rows[0]["name"] == "Test Restaurant"
    assert "Salads" in rows[0]["categories"]
    assert "Healthy" in rows[0]["categories"]
    assert rows[0]["rating"] == 4.5
    assert rows[0]["latitude"] == 40.7
    assert rows[0]["longitude"] == -73.9
    assert rows[0]["search_term"] == "healthy"
    assert rows[0]["anchor_name"] == "downtown"


def test_etl_yelp_extract_businesses_missing_coordinates() -> None:
    from src.data.etl_yelp import _extract_businesses

    businesses = [{"id": "x1", "name": "No Coords", "categories": []}]
    rows = _extract_businesses(businesses, "food", "midtown")
    assert len(rows) == 1
    assert rows[0]["latitude"] is None
    assert rows[0]["longitude"] is None


def test_etl_yelp_extract_businesses_null_coordinates_dict() -> None:
    """Explicitly None coordinates dict should yield None lat/lon."""
    from src.data.etl_yelp import _extract_businesses

    businesses = [
        {
            "id": "y2",
            "name": "Null Coords",
            "categories": [{"title": "Halal"}],
            "coordinates": None,
        }
    ]
    rows = _extract_businesses(businesses, "halal", "bk-tandon")
    assert rows[0]["latitude"] is None
    assert rows[0]["longitude"] is None
    assert rows[0]["categories"] == "Halal"


def test_etl_yelp_extract_businesses_multiple() -> None:
    from src.data.etl_yelp import _extract_businesses

    businesses = [
        {
            "id": "b1",
            "name": "Place A",
            "categories": [{"title": "Indian"}],
            "coordinates": {"latitude": 40.71, "longitude": -73.95},
        },
        {
            "id": "b2",
            "name": "Place B",
            "categories": [],
            "coordinates": {"latitude": 40.72, "longitude": -73.96},
        },
    ]
    rows = _extract_businesses(businesses, "halal", "mn-fidi")
    assert len(rows) == 2
    assert rows[0]["id"] == "b1"
    assert rows[1]["categories"] == ""  # no categories → empty join


# ── etl_inspections — run_etl ─────────────────────────────────────────────────


def test_etl_inspections_run_etl_calls_fetch_and_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.etl_inspections as etl_insp

    sample = pd.DataFrame(
        {
            "inspection_date": ["2024-01-15"],
            "camis": ["12345"],
            "grade": ["A"],
            "critical_flag": ["Not Critical"],
            "boro": ["1"],
            "zipcode": ["10001"],
            "cuisine_description": ["Chinese"],
            "dba": ["Test Restaurant"],
        }
    )
    # Prevent live HTTP call from _get_zip_to_nta
    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN0604"})
    monkeypatch.setattr(etl_insp, "fetch", lambda limit=0: sample)
    result = etl_insp.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "inspection_date" in result.columns
    assert "restaurant_id" in result.columns


# ── etl_licenses — run_etl ────────────────────────────────────────────────────


def test_etl_licenses_run_etl_calls_fetch_and_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.etl_licenses as etl_lic

    sample = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01"],
            "business_unique_id": ["dca-42"],
            "license_status": ["Active"],
            "nta": ["BK09"],
            "business_category": ["Restaurant"],
        }
    )
    monkeypatch.setattr(etl_lic, "fetch", lambda limit=0: sample)
    result = etl_lic.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "event_date" in result.columns
    assert "nta_id" in result.columns


# ── etl_permits — run_etl ─────────────────────────────────────────────────────


def test_etl_permits_run_etl_calls_fetch_and_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.etl_permits as etl_perm

    sample = pd.DataFrame(
        {
            "issueddate": ["2024-01-01"],
            "communityboard": ["101 BROOKLYN"],
            "permitsub": ["NB"],
        }
    )
    monkeypatch.setattr(etl_perm, "fetch", lambda limit=0: sample)
    result = etl_perm.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert "permit_date" in result.columns
    assert "nta_id" in result.columns
    assert "job_count" in result.columns


# ── etl_pluto — run_etl ───────────────────────────────────────────────────────


def test_etl_pluto_run_etl_calls_fetch_and_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.etl_inspections as etl_insp
    import src.data.etl_pluto as etl_pluto

    sample = pd.DataFrame(
        {
            "yearbuilt": ["1990"],
            "zipcode": ["11201"],
            "borough": ["BK"],
            "lotarea": ["5000"],
            "bldgarea": ["3000"],
            "comarea": ["1000"],
            "retailarea": ["500"],
            "assesstot": ["800000"],
        }
    )
    # Prevent live HTTP call from _get_zip_to_nta inside transform
    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"11201": "BK09"})
    monkeypatch.setattr(etl_pluto, "fetch", lambda limit=0: sample)
    result = etl_pluto.run_etl(limit=5)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "nta_id" in result.columns
    assert "commercial_sqft" in result.columns
    assert "assessed_value" in result.columns


# ── etl_runner — strict re-raises FileNotFoundError for non-optional modules ──


def test_etl_runner_strict_reraises_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_runner

    monkeypatch.setattr(
        etl_runner,
        "_ETL_MODULES",
        {
            "inspections": _make_etl_module(
                "inspections", raises=FileNotFoundError("not found")
            ),
        },
    )
    with pytest.raises(FileNotFoundError):
        etl_runner.run_all_etl(strict=True)


# ── quality — prepare_training_frame empty path ───────────────────────────────


def test_prepare_training_frame_empty_input() -> None:
    from src.data.quality import prepare_training_frame

    # Must have target column or ValueError fires before empty-frame logic
    cleaned, report = prepare_training_frame(
        pd.DataFrame({"target": pd.Series([], dtype=float)}),
        target_col="target",
    )
    assert cleaned.empty
    assert report.output_rows == 0


# ── etl_yelp — collect_yelp_businesses ───────────────────────────────────────


def test_collect_yelp_businesses_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_yelp

    monkeypatch.delenv("YELP_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        etl_yelp.collect_yelp_businesses()
    assert excinfo.value.code == 1


def test_collect_yelp_businesses_request_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import requests
    from src.data import etl_yelp

    monkeypatch.setenv("YELP_API_KEY", "fake_key")
    monkeypatch.setattr(
        etl_yelp,
        "ANCHOR_POINTS",
        [{"latitude": 40.7, "longitude": -73.9, "anchor_name": "test"}],
    )
    monkeypatch.setattr(etl_yelp, "SEARCH_TERMS", ["food"])
    monkeypatch.setattr(etl_yelp, "OFFSETS", [0])
    monkeypatch.setattr(etl_yelp, "OUTPUT_PATH", str(tmp_path / "out.csv"))
    monkeypatch.setattr(
        etl_yelp, "SNAPSHOT_PATH_TEMPLATE", str(tmp_path / "snap_{date}.csv")
    )

    def mock_get(*args, **kwargs):
        raise requests.RequestException("API error")

    monkeypatch.setattr(requests, "get", mock_get)

    result = etl_yelp.collect_yelp_businesses()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_collect_yelp_businesses_empty_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import requests
    from src.data import etl_yelp

    monkeypatch.setenv("YELP_API_KEY", "fake_key")
    monkeypatch.setattr(
        etl_yelp,
        "ANCHOR_POINTS",
        [{"latitude": 40.7, "longitude": -73.9, "anchor_name": "test"}],
    )
    monkeypatch.setattr(etl_yelp, "SEARCH_TERMS", ["food"])
    monkeypatch.setattr(etl_yelp, "OFFSETS", [0])
    monkeypatch.setattr(etl_yelp, "OUTPUT_PATH", str(tmp_path / "out.csv"))
    monkeypatch.setattr(
        etl_yelp, "SNAPSHOT_PATH_TEMPLATE", str(tmp_path / "snap_{date}.csv")
    )

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"businesses": []}

    monkeypatch.setattr(requests, "get", lambda *a, **k: MockResponse())

    result = etl_yelp.collect_yelp_businesses()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_collect_yelp_businesses_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import requests
    from src.data import etl_yelp

    monkeypatch.setenv("YELP_API_KEY", "fake_key")
    monkeypatch.setattr(
        etl_yelp,
        "ANCHOR_POINTS",
        [{"latitude": 40.7, "longitude": -73.9, "anchor_name": "test"}],
    )
    monkeypatch.setattr(etl_yelp, "SEARCH_TERMS", ["food"])
    monkeypatch.setattr(etl_yelp, "OFFSETS", [0])
    monkeypatch.setattr(etl_yelp, "OUTPUT_PATH", str(tmp_path / "out.csv"))
    monkeypatch.setattr(
        etl_yelp, "SNAPSHOT_PATH_TEMPLATE", str(tmp_path / "snap_{date}.csv")
    )

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "businesses": [
                    {
                        "id": "b1",
                        "name": "Test Restaurant",
                        "categories": [{"title": "Indian"}],
                        "rating": 4.2,
                        "review_count": 50,
                        "price": "$$",
                        "is_closed": False,
                        "coordinates": {"latitude": 40.7, "longitude": -73.9},
                    }
                ]
            }

    monkeypatch.setattr(requests, "get", lambda *a, **k: MockResponse())
    monkeypatch.setattr(etl_yelp, "REQUEST_SLEEP_SECONDS", 0)

    result = etl_yelp.collect_yelp_businesses()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "id" in result.columns
    assert "name" in result.columns
    assert "rating" in result.columns
    assert result.iloc[0]["id"] == "b1"


# ── etl_acs — NYC NTA profile extract ────────────────────────────────────────


def test_etl_acs_transform_nyc_profile_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.etl_acs import _transform

    monkeypatch.setenv("ACS_YEAR", "2024")
    df = pd.DataFrame({"GeoID": ["BK09"], "Pop16plE": ["5000"], "MdHHIncE": ["75000"]})
    result = _transform(df)
    assert list(result.columns) == [
        "year",
        "nta_id",
        "median_income",
        "population",
        "rent_burden",
    ]
    assert result.iloc[0]["nta_id"] == "BK09"
    assert result.iloc[0]["population"] == 5000
    assert result.iloc[0]["median_income"] == 75000
    # rent_burden estimated as 30% of monthly income (MdHHIncE * 0.30 / 12)
    assert result.iloc[0]["rent_burden"] == pytest.approx(75000 * 0.30 / 12, rel=1e-3)


# ── etl_airbnb — _read_local exception ───────────────────────────────────────


def test_etl_airbnb_read_local_unreadable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_airbnb

    bad_file = tmp_path / "corrupt.csv"
    bad_file.write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV", bad_file)
    monkeypatch.setattr(etl_airbnb, "_RAW_CSV_GZ", tmp_path / "nonexistent.csv.gz")

    def mock_read_csv(*args, **kwargs):
        raise pd.errors.ParserError("Corrupt file")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = etl_airbnb._read_local(limit=50000)
    assert result is None


# ── etl_citibike — LFS pointer detection ─────────────────────────────────────


def test_etl_citibike_run_etl_lfs_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data import etl_citibike

    lfs_file = tmp_path / "citibike.zip"
    lfs_file.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    monkeypatch.setattr(etl_citibike, "_RAW_ZIP", lfs_file)
    # Also clear the fallback CSV so there's no secondary source to fall back to
    monkeypatch.setattr(
        etl_citibike, "_FALLBACK_FEATURES_CSV", tmp_path / "nonexistent.csv"
    )

    def mock_get(*args, **kwargs):
        raise RuntimeError("Abort download after LFS detection")

    monkeypatch.setattr("requests.get", mock_get)

    result = etl_citibike.run_etl()
    assert (
        result.empty
    )  # No local zips + no fallback CSV + failed download → placeholder


# ── etl_licenses — business_unique_id fallback ───────────────────────────────


def test_etl_licenses_transform_adds_business_unique_id_na() -> None:
    from src.data.etl_licenses import transform

    df = pd.DataFrame(
        {
            "license_creation_date": ["2024-01-01"],
            "nta": ["BK09"],
            "business_category": ["Restaurant"],
            "license_status": ["Active"],
        }
    )
    result = transform(df)
    assert "business_unique_id" in result.columns
    assert result.iloc[0]["business_unique_id"] == "UNKNOWN"


# ── etl_pluto — _get_zip_to_nta exception ────────────────────────────────────


def test_etl_pluto_transform_zip_to_nta_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_pluto

    def mock_get_zip_to_nta():
        raise RuntimeError("Mocked exception")

    monkeypatch.setattr("src.data.etl_inspections._get_zip_to_nta", mock_get_zip_to_nta)

    df = pd.DataFrame(
        {
            "yearbuilt": ["1990"],
            "zipcode": ["11201"],
            "comarea": ["5000"],
            "retailarea": ["1000"],
            "lotarea": ["10000"],
            "bldgarea": ["8000"],
            "assesstot": ["1000000"],
        }
    )
    result = etl_pluto.transform(df)
    # Since zip_nta is empty and borough is missing, nta_id should be NaN or empty
    assert result["nta_id"].isna().all()


# ── quality — dedupe_columns fallback ────────────────────────────────────────


def test_prepare_embedding_corpus_dedupe_fallback_after_filtering() -> None:
    from src.data.quality import prepare_embedding_corpus

    df = pd.DataFrame(
        {
            "review_text": [
                "long enough review",
                "long enough review",
                "another long review",
            ]
        }
    )
    # dedupe_columns=['nonexistent'] should filter to [] and
    # then fallback to ['review_text']
    result, report = prepare_embedding_corpus(df, dedupe_columns=["nonexistent"])
    assert len(result) == 2
    assert "long enough review" in result["review_text"].values
    assert "another long review" in result["review_text"].values


def test_etl_acs_transform_invalid_schema():
    from src.data.etl_acs import _transform

    with pytest.raises(ValueError, match="does not match expected schema"):
        _transform(pd.DataFrame({"wrong_col": [1]}))


def test_prepare_embedding_corpus_else_branch_dedupe():
    from src.data.quality import prepare_embedding_corpus

    df = pd.DataFrame({"review_text": ["a" * 10, "b" * 10], "col1": [1, 2]})
    # Hits line 71: dedupe_columns is NOT None
    res, rep = prepare_embedding_corpus(df, dedupe_columns=["col1"])
    assert len(res) == 2


def test_etl_airbnb_run_etl_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_airbnb

    good_result = pd.DataFrame(
        {"nta_id": ["MN01"], "listing_count": [5], "entire_home_ratio": [0.6]}
    )
    monkeypatch.setattr(
        etl_airbnb,
        "_read_local",
        lambda limit: pd.DataFrame({"latitude": [40.7], "longitude": [-74.0]}),
    )
    monkeypatch.setattr(etl_airbnb, "_transform", lambda df: good_result)
    result = etl_airbnb.run_etl(limit=5)
    assert list(result["nta_id"]) == ["MN01"]


# ── etl_citibike — _year_from_zip_name ───────────────────────────────────────


def test_etl_citibike_year_from_zip_name_valid() -> None:
    from pathlib import Path

    from src.data.etl_citibike import _year_from_zip_name

    assert _year_from_zip_name(Path("202603-citibike-tripdata.zip")) == 2026
    assert _year_from_zip_name(Path("202201-citibike-tripdata.zip")) == 2022


def test_etl_citibike_year_from_zip_name_invalid() -> None:
    from pathlib import Path

    from src.data.etl_citibike import _year_from_zip_name

    assert _year_from_zip_name(Path("something.zip")) is None
    assert _year_from_zip_name(Path("citibike-tripdata.zip")) is None


# ── etl_citibike — local glob zip path (lines 138-193) ───────────────────────


def test_etl_citibike_run_etl_local_glob_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import io
    import zipfile

    from src.data import etl_citibike

    # Create a real zip matching *-citibike-tripdata.zip so the glob picks it up
    zip_path = tmp_path / "202603-citibike-tripdata.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_content = (
            "start_lat,start_lng,start_station_id\n"
            "40.75,-73.99,S1\n"
            "40.76,-73.98,S1\n"
            "40.77,-73.97,S2\n"
        )
        zf.writestr("trips.csv", csv_content)
    zip_path.write_bytes(buf.getvalue())

    # _RAW_ZIP.parent must be tmp_path so the glob finds our zip
    monkeypatch.setattr(etl_citibike, "_RAW_ZIP", zip_path)
    monkeypatch.setattr(
        etl_citibike, "_FALLBACK_FEATURES_CSV", tmp_path / "nonexistent.csv"
    )
    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["MN2601"] * len(lat), index=lat.index),
    )

    result = etl_citibike.run_etl(limit=100)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"year", "nta_id", "trip_count", "station_count"}
    assert len(result) > 0
    assert (result["trip_count"] > 0).all()


def test_etl_citibike_run_etl_local_glob_with_backfill(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Backfill CSV supplements missing years when not all years are in the zip data."""
    import io
    import zipfile

    from src.data import etl_citibike

    # Zip only covers 2026 trips
    zip_path = tmp_path / "202603-citibike-tripdata.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "trips.csv",
            "start_lat,start_lng\n40.75,-73.99\n",
        )
    zip_path.write_bytes(buf.getvalue())

    # Fallback CSV has historical NTA data for backfill
    fallback_csv = tmp_path / "citibike_fallback.csv"
    fallback_csv.write_text("nta,trip_count,unique_start_station_count\nBK09,50,3\n")

    monkeypatch.setattr(etl_citibike, "_RAW_ZIP", zip_path)
    monkeypatch.setattr(etl_citibike, "_FALLBACK_FEATURES_CSV", fallback_csv)
    monkeypatch.setattr(
        "src.utils.geospatial.lat_lon_to_nta",
        lambda lat, lon: pd.Series(["MN2601"] * len(lat), index=lat.index),
    )
    # Force a wide year range so backfill is needed
    monkeypatch.setitem(
        __import__("src.config.constants", fromlist=["MODEL_CONFIG"]).MODEL_CONFIG,
        "temporal_data_start_year",
        2020,
    )
    monkeypatch.setitem(
        __import__("src.config.constants", fromlist=["MODEL_CONFIG"]).MODEL_CONFIG,
        "temporal_data_end_year",
        2022,
    )

    result = etl_citibike.run_etl(limit=100)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"year", "nta_id", "trip_count", "station_count"}
    # Backfill should produce rows for 2020, 2021, 2022
    years_present = set(result["year"].unique())
    assert years_present.issuperset({2020, 2021, 2022})


# ── etl_acs — ACS_DATA_GLOB branch (lines 50-61) ─────────────────────────────


def test_etl_acs_load_local_glob_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data.etl_acs import _load_local

    csv1 = tmp_path / "acs_2022.csv"
    csv2 = tmp_path / "acs_2023.csv"
    csv1.write_text(
        "year,nta_id,median_income,population,rent_burden\n"
        "2022,BK0900,60000,5000,1500\n"
    )
    csv2.write_text(
        "year,nta_id,median_income,population,rent_burden\n"
        "2023,BK0900,62000,5100,1550\n"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ACS_DATA_GLOB", "acs_*.csv")
    monkeypatch.delenv("ACS_DATA_PATH", raising=False)

    result = _load_local()
    assert len(result) == 2
    assert set(result["year"].tolist()) == {2022, 2023}


def test_etl_acs_load_local_glob_no_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.data.etl_acs import _load_local

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ACS_DATA_GLOB", "nonexistent_acs_*.csv")
    monkeypatch.delenv("ACS_DATA_PATH", raising=False)

    with pytest.raises(FileNotFoundError, match="ACS_DATA_GLOB"):
        _load_local()


def test_etl_acs_run_etl_no_canonical_file_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Lines 145-146: no env var + canonical CSV absent → placeholder."""
    from src.data import etl_acs

    # cwd has no data/raw/acs_nta_canonical.csv so canonical_path.exists() is False
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACS_DATA_PATH", raising=False)
    monkeypatch.delenv("ACS_DATA_GLOB", raising=False)

    result = etl_acs.run_etl(limit=5)
    assert result.empty


# ── etl_yelp — _env_float and _env_int valid-value paths (lines 40-43, 50-53)


def test_etl_yelp_env_float_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lines 40-43: _env_float returns parsed float when env var is a valid number."""
    from src.data.etl_yelp import _env_float

    monkeypatch.setenv("YELP_TEST_FLOAT", "0.05")
    result = _env_float("YELP_TEST_FLOAT", 0.03)
    assert result == pytest.approx(0.05)


def test_etl_yelp_env_float_invalid_value_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.etl_yelp import _env_float

    monkeypatch.setenv("YELP_TEST_FLOAT", "not_a_float")
    result = _env_float("YELP_TEST_FLOAT", 0.03)
    assert result == pytest.approx(0.03)


def test_etl_yelp_env_int_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lines 50-53: _env_int returns parsed int when env var is a valid integer."""
    from src.data.etl_yelp import _env_int

    monkeypatch.setenv("YELP_TEST_INT", "42")
    result = _env_int("YELP_TEST_INT", 10)
    assert result == 42


def test_etl_yelp_env_int_invalid_value_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.etl_yelp import _env_int

    monkeypatch.setenv("YELP_TEST_INT", "not_an_int")
    result = _env_int("YELP_TEST_INT", 10)
    assert result == 10


def test_etl_yelp_load_local_from_env_path_additional(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Lines 40-43 (desc): Test _load_local when YELP_DATA_PATH is a valid file."""
    from pathlib import Path
    from src.data import etl_yelp

    # Mock default path to not exist
    monkeypatch.setattr(
        etl_yelp, "_DEFAULT_FUSION_REVIEW_PATH", Path("/nonexistent/reviews_v2.csv")
    )

    # Create a temporary CSV file
    csv_path = tmp_path / "env_yelp_data_new.csv"
    pd.DataFrame({"business_id": ["env_r2"], "rating": [5]}).to_csv(
        csv_path, index=False
    )

    monkeypatch.setenv("YELP_DATA_PATH", str(csv_path))
    result = etl_yelp._load_local()
    assert not result.empty
    assert result.iloc[0]["business_id"] == "env_r2"


def test_etl_yelp_env_helpers_additional_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 50-53 (desc): Test _env_int helper's integer parsing branches."""
    from src.data.etl_yelp import _env_int

    # Test valid integer parsing (Line 50-51)
    monkeypatch.setenv("YELP_TEST_INT_VAL", "200")
    assert _env_int("YELP_TEST_INT_VAL", 0) == 200

    # Test invalid integer parsing (Line 52-53)
    monkeypatch.setenv("YELP_TEST_INT_VAL", "not_an_int")
    assert _env_int("YELP_TEST_INT_VAL", 50) == 50


# ── etl_pluto (Added tests) ──────────────────────────────────────────────────


def test_etl_pluto_transform_renames_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"10001": "MN0604"})
    raw = pd.DataFrame(
        {
            "zipcode": ["10001"],
            "borough": ["MN"],
            "lotarea": ["1000"],
            "bldgarea": ["1000"],
            "comarea": ["500"],
            "retailarea": ["100"],
            "assesstot": ["100000"],
        }
    )
    transformed = etl_pluto.transform(raw)
    assert "commercial_sqft" in transformed.columns
    assert "assessed_value" in transformed.columns
    assert "comarea" not in transformed.columns
    assert "assesstot" not in transformed.columns


def test_etl_pluto_transform_drops_unmapped_nta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    # Mock only 11201 as mapped
    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"11201": "BK09"})
    raw = pd.DataFrame(
        {
            "zipcode": ["11201", "99999"],  # 11201 mapped, 99999 unmapped
            "borough": ["BK", "MN"],
            "lotarea": ["1000", "1000"],
            "bldgarea": ["1000", "1000"],
            "comarea": ["500", "500"],
            "retailarea": ["100", "100"],
            "assesstot": ["100000", "100000"],
        }
    )
    transformed = etl_pluto.transform(raw)
    # Only 1 NTA in output
    assert len(transformed["nta_id"].unique()) == 1
    assert transformed["nta_id"].unique()[0] == "BK09"


def test_etl_pluto_transform_aggregates_by_nta(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"11201": "BK09"})
    raw = pd.DataFrame(
        {
            "zipcode": ["11201", "11201"],
            "borough": ["BK", "BK"],
            "lotarea": ["1000", "1000"],
            "bldgarea": ["1000", "1000"],
            "comarea": ["500", "500"],  # sum = 1000
            "retailarea": ["100", "100"],
            "assesstot": ["100000", "200000"],  # sum = 300000
        }
    )
    transformed = etl_pluto.transform(raw)
    # Filter for one year, e.g., 2020
    row = transformed[transformed["year"] == 2020].iloc[0]
    assert row["commercial_sqft"] == 1000
    assert row["assessed_value"] == 300000


def test_etl_pluto_transform_expands_years(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"11201": "BK09"})
    raw = pd.DataFrame(
        {
            "zipcode": ["11201"],
            "borough": ["BK"],
            "lotarea": ["1000"],
            "bldgarea": ["1000"],
            "comarea": ["500"],
            "retailarea": ["100"],
            "assesstot": ["100000"],
        }
    )
    transformed = etl_pluto.transform(raw)
    assert set(transformed["year"].unique()) == {2020, 2021, 2022, 2023, 2024}


def test_etl_pluto_transform_produces_correct_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import etl_pluto
    import src.data.etl_inspections as etl_insp

    monkeypatch.setattr(etl_insp, "_ZIP_TO_NTA", {"11201": "BK09", "10001": "MN06"})
    raw = pd.DataFrame(
        {
            "zipcode": ["11201", "10001"],
            "borough": ["BK", "MN"],
            "lotarea": ["1000", "1000"],
            "bldgarea": ["1000", "1000"],
            "comarea": ["500", "500"],
            "retailarea": ["100", "100"],
            "assesstot": ["100000", "100000"],
        }
    )
    transformed = etl_pluto.transform(raw)
    # 2 NTAs * 5 years = 10 rows
    assert len(transformed) == 10


def test_etl_pluto_fetch_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from src.data import etl_pluto

    def mock_get(*args, **kwargs):
        resp = requests.Response()
        resp.status_code = 500
        return resp

    monkeypatch.setattr(requests, "get", mock_get)
    with pytest.raises(requests.exceptions.HTTPError):
        etl_pluto.fetch()
