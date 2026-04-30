"""Map view — renders micro-zone recommendation scores on a Plotly scatter map."""

from __future__ import annotations

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


def render_map_view(recommendations: list[dict] | None = None) -> None:
    """Render micro-zone opportunity scores on a map.

    If recommendations are provided, plots each scored zone with color
    proportional to opportunity score. Falls back to plain NTA centroids
    (or a placeholder) when no recommendations are available.
    """
    st.subheader("Zone Map")
    st.caption("See where the strongest zones cluster. Hover for score, risk, and confidence.")
    if recommendations:
        try:
            import plotly.graph_objects as go

            lats, lons, texts, scores, zone_ids = [], [], [], [], []
            for rec in recommendations:
                zid = rec.get("zone_id", "")
                coords = _ZONE_COORDS.get(zid)
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
                            size=18,
                            color=scores,
                            colorscale="RdYlGn",
                            cmin=0,
                            cmax=100,
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
        except Exception:
            pass  # fall through to plain map

    # Plain NTA centroid fallback
    if _GEOJSON_PATH.exists():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(str(_GEOJSON_PATH))
            centroids = gdf.copy()
            centroids["geometry"] = centroids.geometry.centroid
            centroids["lat"] = centroids.geometry.y
            centroids["lon"] = centroids.geometry.x
            points_df = centroids[["lat", "lon"]].dropna()
            st.map(points_df, zoom=10)
            return
        except Exception:
            pass

    st.info("Map: micro-zone geometries render here once boundary data is loaded.")
