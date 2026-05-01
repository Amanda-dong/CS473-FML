"""Map view — plots NTA recommendations using borough centroids."""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from frontend.components.recommendation_card import _display_name

BOROUGH_CENTROIDS = {
    "BK": (40.6501, -73.9496),
    "QN": (40.7282, -73.7949),
    "MN": (40.7831, -73.9712),
    "BX": (40.8448, -73.8648),
    "SI": (40.5795, -74.1502),
}

BOROUGH_PREFIX = {
    "Brooklyn": "BK",
    "Queens": "QN",
    "Manhattan": "MN",
    "Bronx": "BX",
    "Staten Island": "SI",
}

MARKET_TYPE_COLOR = {
    "High Opportunity": "red",
    "Established Hub": "blue",
    "Growing Market": "green",
    "Low Demand": "gray",
}


def _deterministic_jitter(nta_id: str, base_lat: float, base_lon: float) -> tuple[float, float]:
    seed = sum(ord(c) for c in nta_id)
    rng = random.Random(seed)
    return (
        base_lat + rng.uniform(-0.03, 0.03),
        base_lon + rng.uniform(-0.04, 0.04),
    )


def render_map_view(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No data to display on map.")
        return

    st.subheader("Neighborhood Map")
    st.caption(
        "Use this map to see where your strongest matches cluster. Marker positions are approximate, not exact neighborhood boundaries."
    )

    # Build points dataframe
    rows = []
    for _, row in df.iterrows():
        nta_id = str(row.get("nta_id", ""))
        prefix = nta_id[:2].upper()
        base_coords = BOROUGH_CENTROIDS.get(prefix)
        if base_coords is None:
            continue
        lat, lon = _deterministic_jitter(nta_id, base_coords[0], base_coords[1])
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "nta_id": nta_id,
                "label": _display_name(nta_id),
                "market_type": str(row.get("market_type", "")),
                "final_score": float(row.get("final_score", 0.0)),
                "risk_bucket": str(row.get("risk_bucket", "")),
            }
        )

    if not rows:
        st.info("No mappable NTAs found.")
        return

    points_df = pd.DataFrame(rows)
    points_df["marker_size"] = 12

    try:
        import plotly.express as px

        borough_zoom = {
            "Brooklyn": (40.6501, -73.9496, 12),
            "Queens": (40.7282, -73.7949, 11),
            "Manhattan": (40.7831, -73.9712, 12),
            "Bronx": (40.8448, -73.8648, 12),
            "Staten Island": (40.5795, -74.1502, 12),
        }

        selected_borough = st.selectbox(
            "Map focus",
            ["Citywide", "Brooklyn", "Queens", "Manhattan", "Bronx", "Staten Island"],
            help="Center the map on one borough while keeping your current results visible.",
        )

        center_lat, center_lon, zoom = 40.730, -73.935, 10
        map_df = points_df
        if selected_borough != "Citywide":
            center_lat, center_lon, zoom = borough_zoom[selected_borough]
            borough_prefix = BOROUGH_PREFIX[selected_borough]
            map_df = points_df[points_df["nta_id"].str.startswith(borough_prefix)].copy()

        if map_df.empty:
            st.info("No current results are shown in that borough.")
            return

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="market_type",
            color_discrete_map=MARKET_TYPE_COLOR,
            size="marker_size",
            size_max=18,
            hover_name="label",
            hover_data={
                "nta_id": True,
                "final_score": ":.3f",
                "risk_bucket": True,
                "market_type": True,
                "lat": False,
                "lon": False,
            },
            zoom=10,
            height=450,
            title="Top neighborhoods by score",
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
            margin=dict(l=0, r=0, t=40, b=0),
            legend_title_text="Area type",
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to st.map if plotly not available
        st.map(points_df[["lat", "lon"]], zoom=10)

    # Legend
    st.caption(
        "🔴 High Opportunity  ·  🔵 Established Hub  ·  "
        "🟢 Growing Market  ·  🔘 Low Demand"
    )
