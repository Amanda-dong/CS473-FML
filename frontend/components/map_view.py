"""Map view — plots NTA recommendations using borough centroids."""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

BOROUGH_CENTROIDS = {
    "BK": (40.6501, -73.9496),
    "QN": (40.7282, -73.7949),
    "MN": (40.7831, -73.9712),
    "BX": (40.8448, -73.8648),
    "SI": (40.5795, -74.1502),
}

MARKET_TYPE_COLOR = {
    "High Opportunity": "red",
    "Established Hub": "blue",
    "Growing Market": "green",
    "Low Demand": "gray",
}


def _jitter(val: float, amount: float = 0.018) -> float:
    return val + random.uniform(-amount, amount)


def render_map_view(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No data to display on map.")
        return

    st.subheader("NTA Locations")
    st.caption(
        "Markers are approximate borough-level centroids with jitter. "
        "Exact NTA boundaries require GeoJSON data."
    )

    # Build points dataframe
    rows = []
    for _, row in df.iterrows():
        nta_id = str(row.get("nta_id", ""))
        prefix = nta_id[:2].upper()
        base_coords = BOROUGH_CENTROIDS.get(prefix)
        if base_coords is None:
            continue
        rows.append(
            {
                "lat": _jitter(base_coords[0]),
                "lon": _jitter(base_coords[1]),
                "nta_id": nta_id,
                "market_type": str(row.get("market_type", "")),
                "final_score": float(row.get("final_score", 0.0)),
                "risk_bucket": str(row.get("risk_bucket", "")),
            }
        )

    if not rows:
        st.info("No mappable NTAs found.")
        return

    points_df = pd.DataFrame(rows)

    try:
        import plotly.express as px

        fig = px.scatter_mapbox(
            points_df,
            lat="lat",
            lon="lon",
            color="market_type",
            color_discrete_map=MARKET_TYPE_COLOR,
            hover_name="nta_id",
            hover_data={
                "final_score": ":.3f",
                "risk_bucket": True,
                "market_type": True,
                "lat": False,
                "lon": False,
            },
            zoom=10,
            height=450,
            title="Top NTAs by opportunity score",
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=40.730, lon=-73.935)),
            margin=dict(l=0, r=0, t=40, b=0),
            legend_title_text="Market Type",
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to st.map if plotly not available
        st.map(points_df[["lat", "lon"]], zoom=10)

    # Legend
    st.caption(
        "🔴 High Opportunity  ·  🔵 Established Hub  ·  "
        "🟢 Growing Market  ·  ⚫ Low Demand"
    )
