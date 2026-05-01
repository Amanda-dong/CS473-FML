import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.utils.geospatial import _load_nta_polygons, lat_lon_to_nta


@pytest.fixture(autouse=True)
def clear_cache():
    _load_nta_polygons.cache_clear()


def test_load_nta_polygons_shapely_import_error(monkeypatch):
    with patch.dict(sys.modules, {"shapely": None}):
        codes, tree = _load_nta_polygons()
        assert codes == []
        assert tree is None


def test_load_nta_polygons_no_files_found(monkeypatch):
    monkeypatch.setattr(
        "src.utils.geospatial._NTA_GEOJSON_CANDIDATES", ["nonexistent.geojson"]
    )
    codes, tree = _load_nta_polygons()
    assert codes == []
    assert tree is None


def test_load_nta_polygons_corrupt_json(monkeypatch, tmp_path):
    f = tmp_path / "corrupt.geojson"
    f.write_text("{ corrupt }")
    monkeypatch.setattr("src.utils.geospatial._NTA_GEOJSON_CANDIDATES", [str(f)])
    codes, tree = _load_nta_polygons()
    assert codes == []
    assert tree is None


def test_load_nta_polygons_skips_empty_nta2020(monkeypatch, tmp_path):
    f = tmp_path / "test.geojson"
    data = {
        "features": [
            {
                "properties": {"nta2020": ""},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            },
            {
                "properties": {"nta2020": "NTA1"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            },
        ]
    }
    f.write_text(json.dumps(data))
    monkeypatch.setattr("src.utils.geospatial._NTA_GEOJSON_CANDIDATES", [str(f)])
    codes, tree = _load_nta_polygons()
    assert "NTA1" in codes
    assert "" not in codes


def test_load_nta_polygons_handles_shape_exception(monkeypatch, tmp_path):
    f = tmp_path / "test.geojson"
    # Use geometry that shapely.geometry.shape() fails on
    data = {
        "features": [
            {
                "properties": {"nta2020": "BAD"},
                "geometry": {"type": "Point", "coordinates": "invalid"},
            },
        ]
    }
    f.write_text(json.dumps(data))
    monkeypatch.setattr("src.utils.geospatial._NTA_GEOJSON_CANDIDATES", [str(f)])
    codes, tree = _load_nta_polygons()
    assert codes == []


def test_lat_lon_to_nta_no_codes(monkeypatch):
    monkeypatch.setattr("src.utils.geospatial._load_nta_polygons", lambda: ([], None))
    lat = pd.Series([40.7], index=[0])
    lon = pd.Series([-74.0], index=[0])
    result = lat_lon_to_nta(lat, lon)
    assert result.iloc[0] == ""


def test_lat_lon_to_nta_centroid_fallback_exception(monkeypatch):
    mock_tree = MagicMock()
    mock_tree.query.return_value = (np.array([], dtype=int), np.array([], dtype=int))
    mock_tree.geometries = np.array([])

    monkeypatch.setattr(
        "src.utils.geospatial._load_nta_polygons", lambda: (["NTA1"], mock_tree)
    )

    with patch("shapely.centroid", side_effect=Exception("Boom")):
        lat = pd.Series([40.7], index=[0])
        lon = pd.Series([-74.0], index=[0])
        result = lat_lon_to_nta(lat, lon)
        # When fallback fails, it leaves result at initial value ""
        assert result.iloc[0] == ""
