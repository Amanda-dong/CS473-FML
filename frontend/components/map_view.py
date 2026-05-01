"""Map view — renders micro-zone recommendation scores on a Plotly scatter map."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_GEOJSON_PATH = Path("data/geojson/nta_boundaries.geojson")

_ZONE_COORDS: dict[str, tuple[float, float]] = {
    "bk-tandon": (40.6928, -73.9872),
    "bk-downtownbk": (40.6935, -73.9866),
    "bk-williamsburg": (40.7136, -73.9537),
    "bk-navy-yard": (40.6995, -73.9710),
    "bk-fort-greene": (40.6891, -73.9744),
    "bk-crown-hts": (40.6681, -73.9442),
    "bk-sunset-pk": (40.6451, -74.0020),
    "mn-midtown-e": (40.7549, -73.9730),
    "mn-fidi": (40.7074, -74.0113),
    "mn-columbia": (40.8075, -73.9626),
    "mn-nyu-wash-sq": (40.7295, -73.9965),
    "mn-ues-hosp": (40.7701, -73.9547),
    "mn-chelsea": (40.7465, -74.0014),
    "mn-harlem": (40.8116, -73.9465),
    "mn-lic-adj": (40.7529, -73.9677),
    "qn-lic": (40.7471, -73.9440),
    "qn-astoria": (40.7722, -73.9301),
    "qn-flushing": (40.7675, -73.8330),
    "qn-jackson-hts": (40.7498, -73.8830),
    "qn-forest-hills": (40.7196, -73.8448),
    "qn-jamaica": (40.6921, -73.8063),
    "bx-fordham": (40.8609, -73.8896),
    "bx-mott-haven": (40.8084, -73.9218),
    "bx-co-op-city": (40.8743, -73.8296),
    "si-st-george": (40.6437, -74.0733),
}


def _geometry_centroid(geometry: dict) -> tuple[float, float] | None:
    """Compute centroid from GeoJSON Polygon/MultiPolygon coordinates."""
    geom_type = str(geometry.get("type", ""))
    coords = geometry.get("coordinates")
    if not coords:
        return None

    points: list[tuple[float, float]] = []
    polygons = [coords] if geom_type == "Polygon" else coords if geom_type == "MultiPolygon" else []
    for poly in polygons:
        if not poly or not poly[0]:
            continue
        for point in poly[0]:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            points.append((float(point[1]), float(point[0])))
    if not points:
        return None
    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    return (lat, lon)


@st.cache_data(show_spinner=False)
def _build_recommended_zone_coords(zone_ids: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """Build coords for only requested zone_ids via zone->NTA crosswalk."""
    if not zone_ids or not _GEOJSON_PATH.exists():
        return {}
    try:
        from src.features.zone_crosswalk import ZONE_TO_NTA

        payload = json.loads(_GEOJSON_PATH.read_text(encoding="utf-8"))
        nta_centroids: dict[str, tuple[float, float]] = {}
        for feature in payload.get("features", []):
            props = feature.get("properties", {}) if isinstance(feature, dict) else {}
            nta = str(props.get("nta2020", "")).strip().upper()
            if not nta:
                continue
            centroid = _geometry_centroid(feature.get("geometry", {}) or {})
            if centroid is not None:
                nta_centroids[nta] = centroid

        zone_coords: dict[str, tuple[float, float]] = {}
        for zone_id in zone_ids:
            nta_list = ZONE_TO_NTA.get(zone_id, [])
            pts = [nta_centroids[nta] for nta in nta_list if nta in nta_centroids]
            if not pts:
                continue
            zone_coords[zone_id] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
        return zone_coords
    except (ValueError, KeyError, TypeError, ImportError, OSError, json.JSONDecodeError):
        return {}


def render_map_view(recommendations: list[dict] | None = None) -> None:
    """Render micro-zone opportunity scores on a map.

    If recommendations are provided, plots each scored zone with color
    proportional to opportunity score. Falls back to plain NTA centroids
    (or a placeholder) when no recommendations are available.
    """
    st.subheader("Zone Map")
    st.caption(
        "See where the strongest zones cluster. Hover for score, risk, and confidence."
    )
    if recommendations:
        try:
            import plotly.graph_objects as go

            recommended_ids = tuple(
                str(rec.get("zone_id", "")) for rec in recommendations if rec.get("zone_id")
            )
            dynamic_coords = _build_recommended_zone_coords(recommended_ids)
            lats, lons, texts, scores, zone_ids = [], [], [], [], []
            for rec in recommendations:
                zid = rec.get("zone_id", "")
                coords = _ZONE_COORDS.get(zid) or dynamic_coords.get(str(zid))
                if coords is None:
                    continue
                lats.append(coords[0])
                lons.append(coords[1])
                score = float(rec.get("opportunity_score", 0.0) or 0.0)
                scores.append(score * 100)
                name = rec.get("zone_name", zid)
                risk = float(rec.get("survival_risk", 0.0) or 0.0)
                conf = str(rec.get("confidence_bucket", "—")).title()
                texts.append(
                    f"<b>{name}</b><br>Score: {score * 100:.0f}%<br>Risk: {risk * 100:.0f}%<br>Confidence: {conf}"
                )
                zone_ids.append(zid)

            if lats:
                fig = go.Figure(
                    go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode="markers",
                        marker=go.scattermapbox.Marker(
                            size=13,
                            color=scores,
                            colorscale="Viridis",
                            cmin=0,
                            cmax=100,
                            opacity=0.95,
                            colorbar=dict(title="Score %", thickness=12),
                        ),
                        text=texts,
                        hoverinfo="text",
                    )
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox=dict(center=dict(lat=40.730, lon=-73.935), zoom=10),
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                return
        except (ValueError, TypeError, ImportError):
            pass  # fall through to plain map

    st.info("No recommendation points to display for the current query.")
