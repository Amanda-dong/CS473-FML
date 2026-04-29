"""Build NTA-level feature tables for Yelp, hygiene, census, and Citi Bike."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
NTA_GEOJSON_PATH = RAW_DIR / "nta.geojson"
OLD_NTA_GDB_PATH = RAW_DIR / "acs_nta_2014_2018" / "NTA_ACS_2014_2018.gdb"
YELP_PATH = RAW_DIR / "yelp_business.csv"
HYGIENE_PATH = RAW_DIR / "restaurant_hygiene.csv"
CENSUS_PATH = RAW_DIR / "census_nta.csv"
CITIBIKE_GLOB = "citibike_202603/*.csv"


def load_manhattan_ntas() -> gpd.GeoDataFrame:
    """Backward-compatible entry returning Manhattan NTAs only."""

    return _load_ntas(borough="manhattan")


def _load_ntas(*, borough: str | None = None) -> gpd.GeoDataFrame:
    new_nta = gpd.read_file(NTA_GEOJSON_PATH)
    new_nta["boroname"] = new_nta["boroname"].astype(str)
    if borough:
        new_nta = new_nta[new_nta["boroname"].str.lower() == borough.lower()].copy()
    new_nta = new_nta[["nta2020", "ntaname", "geometry"]]

    # Preferred path: map 2020 NTAs to legacy ACS NTAs via centroid spatial join.
    # Some local setups do not include the old ACS geodatabase; in that case,
    # gracefully fall back to using NTA2020 codes directly as ``nta``.
    if not OLD_NTA_GDB_PATH.exists():
        fallback = new_nta.copy()
        fallback["nta"] = fallback["nta2020"]
        fallback["nta_name"] = fallback["ntaname"]
        return fallback[["nta2020", "nta", "nta_name", "geometry"]]

    old_nta = gpd.read_file(OLD_NTA_GDB_PATH, layer="NTA_ACS_Demographics")
    old_nta["BoroName"] = old_nta["BoroName"].astype(str)
    if borough:
        old_nta = old_nta[old_nta["BoroName"].str.lower() == borough.lower()].copy()
    old_nta = old_nta[["NTACode", "NTAName", "geometry"]]

    new_proj = new_nta.to_crs(2263)
    old_proj = old_nta.to_crs(2263)
    centroid_points = gpd.GeoDataFrame(
        new_proj[["nta2020", "ntaname"]].copy(),
        geometry=new_proj.geometry.centroid,
        crs=new_proj.crs,
    )
    crosswalk = gpd.sjoin(
        centroid_points,
        old_proj[["NTACode", "NTAName", "geometry"]],
        how="left",
        predicate="within",
    )[["nta2020", "ntaname", "NTACode", "NTAName"]]

    merged = new_nta.merge(crosswalk, on=["nta2020", "ntaname"], how="left")
    merged = merged.rename(columns={"NTACode": "nta", "NTAName": "nta_name"})
    merged = merged.dropna(subset=["nta"]).copy()
    return merged[["nta2020", "nta", "nta_name", "geometry"]]


def load_ntas() -> gpd.GeoDataFrame:
    """Public entry: all NYC NTA polygons with ACS ``nta`` codes."""

    return _load_ntas()


def _spatial_join_points(
    frame: pd.DataFrame,
    *,
    lat_col: str,
    lng_col: str,
    nta_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    usable = frame.dropna(subset=[lat_col, lng_col]).copy()
    point_gdf = gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable[lng_col], usable[lat_col]),
        crs="EPSG:4326",
    )
    return gpd.sjoin(
        point_gdf, nta_gdf[["nta", "geometry"]], how="inner", predicate="within"
    )


def build_yelp_features(nta_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    yelp = pd.read_csv(YELP_PATH)
    yelp["name"] = yelp["name"].fillna("")
    yelp["categories"] = yelp["categories"].fillna("")
    yelp["rating"] = pd.to_numeric(yelp["rating"], errors="coerce")
    yelp["review_count"] = pd.to_numeric(yelp["review_count"], errors="coerce").fillna(
        0
    )
    yelp["is_halal"] = (
        yelp["name"].str.lower().str.contains("halal", na=False)
        | yelp["categories"].str.lower().str.contains("halal", na=False)
    ).astype(int)

    joined = _spatial_join_points(
        yelp, lat_col="latitude", lng_col="longitude", nta_gdf=nta_gdf
    )
    features = (
        joined.groupby("nta", as_index=False)
        .agg(
            restaurant_count=("id", "count"),
            halal_count=("is_halal", "sum"),
            avg_rating=("rating", "mean"),
            total_review_count=("review_count", "sum"),
        )
        .sort_values("nta")
        .reset_index(drop=True)
    )
    features["halal_share"] = features["halal_count"] / features["restaurant_count"]
    return features[
        [
            "nta",
            "restaurant_count",
            "halal_count",
            "halal_share",
            "avg_rating",
            "total_review_count",
        ]
    ]


def build_hygiene_features() -> pd.DataFrame:
    hygiene = pd.read_csv(HYGIENE_PATH)
    hygiene["inspection_date"] = pd.to_datetime(
        hygiene["INSPECTION DATE"], errors="coerce"
    )
    hygiene["nta"] = hygiene["NTA"].fillna("").astype(str).str.strip()
    hygiene = hygiene[hygiene["nta"] != ""].copy()
    hygiene["SCORE"] = pd.to_numeric(hygiene["SCORE"], errors="coerce")
    hygiene["is_critical"] = (
        hygiene["CRITICAL FLAG"].fillna("").str.lower().eq("critical").astype(int)
    )

    features = (
        hygiene.groupby("nta", as_index=False)
        .agg(
            inspection_count=("CAMIS", "count"),
            avg_score=("SCORE", "mean"),
            critical_violation_rate=("is_critical", "mean"),
        )
        .sort_values("nta")
        .reset_index(drop=True)
    )
    return features


def build_census_features() -> pd.DataFrame:
    census = pd.read_csv(CENSUS_PATH)
    nta_crosswalk = load_ntas()[["nta2020", "nta"]].drop_duplicates()
    census = census.merge(
        nta_crosswalk, left_on="GeoID", right_on="nta2020", how="left"
    )
    census = census.dropna(subset=["nta"]).copy()
    features = census[["nta", "MdHHIncE", "Pop16plE"]].rename(
        columns={
            "MdHHIncE": "median_household_income",
            "Pop16plE": "population_16plus",
        }
    )
    features = (
        features.groupby("nta", as_index=False)
        .agg(
            median_household_income=("median_household_income", "mean"),
            population_16plus=("population_16plus", "sum"),
        )
        .sort_values("nta")
        .reset_index(drop=True)
    )
    return features


def build_citibike_features(nta_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    trip_counts: dict[str, int] = {}
    station_sets: dict[str, set[str]] = {}

    for csv_path in sorted(RAW_DIR.glob(CITIBIKE_GLOB)):
        for chunk in pd.read_csv(csv_path, chunksize=200_000, low_memory=False):
            joined = _spatial_join_points(
                chunk, lat_col="start_lat", lng_col="start_lng", nta_gdf=nta_gdf
            )
            if joined.empty:
                continue

            counts = joined.groupby("nta").size()
            for nta, count in counts.items():
                trip_counts[nta] = trip_counts.get(nta, 0) + int(count)

            for nta, stations in joined.groupby("nta")["start_station_id"]:
                station_sets.setdefault(nta, set()).update(
                    str(station) for station in stations.dropna().astype(str)
                )

    rows = []
    for nta in sorted(trip_counts):
        rows.append(
            {
                "nta": nta,
                "trip_count": trip_counts[nta],
                "unique_start_station_count": len(station_sets.get(nta, set())),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["nta", "trip_count", "unique_start_station_count"])
    return pd.DataFrame(rows)


def write_output(frame: pd.DataFrame, output_path: Path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"{output_path}: shape={frame.shape}")
    print(frame.head().to_string(index=False))
    print()


def main() -> None:
    nta_gdf = load_ntas()

    yelp_features = build_yelp_features(nta_gdf)
    write_output(yelp_features, PROCESSED_DIR / "yelp_nta_features.csv")

    hygiene_features = build_hygiene_features()
    write_output(hygiene_features, PROCESSED_DIR / "hygiene_nta_features.csv")

    census_features = build_census_features()
    write_output(census_features, PROCESSED_DIR / "census_nta_features.csv")

    citibike_features = build_citibike_features(nta_gdf)
    write_output(citibike_features, PROCESSED_DIR / "citibike_nta_features.csv")


if __name__ == "__main__":
    main()
