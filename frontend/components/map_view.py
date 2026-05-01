"""Map view — plots NTA recommendations using borough centroids."""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from frontend.components.recommendation_card import _display_name, _load_nta_zone_lookup

BOROUGH_CENTROIDS = {
    "BK": (40.6501, -73.9496),
    "QN": (40.7282, -73.7949),
    "MN": (40.7831, -73.9712),
    "BX": (40.8448, -73.8648),
    "SI": (40.5795, -74.1502),
}

PREFIX_TO_BOROUGH = {
    "BK": "Brooklyn",
    "QN": "Queens",
    "MN": "Manhattan",
    "BX": "Bronx",
    "SI": "Staten Island",
}

MARKET_TYPE_COLOR = {
    "High Opportunity": "red",
    "Established Hub": "blue",
    "Growing Market": "green",
    "Low Demand": "gray",
}

# Approximate zone centers (kept inland) to avoid markers drifting into water.
# These are only used when we can map an NTA to a known zone_id.
ZONE_CENTERS = {
    "bk-tandon": (40.6940, -73.9857),
    "bk-downtownbk": (40.6903, -73.9881),
    "bk-williamsburg": (40.7081, -73.9571),
    "bk-navy-yard": (40.6982, -73.9707),
    "bk-fort-greene": (40.6888, -73.9730),
    "bk-crown-hts": (40.6681, -73.9448),
    "bk-sunset-pk": (40.6455, -74.0124),
    "mn-midtown-e": (40.7549, -73.9715),
    "mn-fidi": (40.7075, -74.0113),
    "mn-columbia": (40.8075, -73.9626),
    "mn-nyu-wash-sq": (40.7308, -73.9973),
    "mn-ues-hosp": (40.7679, -73.9562),
    "mn-chelsea": (40.7465, -74.0014),
    "mn-harlem": (40.8116, -73.9465),
    "mn-lic-adj": (40.7542, -73.9686),
    "qn-lic": (40.7447, -73.9485),
    "qn-astoria": (40.7644, -73.9235),
    "qn-flushing": (40.7654, -73.8174),
    "qn-jackson-hts": (40.7557, -73.8831),
    "qn-forest-hills": (40.7181, -73.8448),
    "qn-jamaica": (40.7027, -73.7890),
    "bx-fordham": (40.8625, -73.8900),
    "bx-mott-haven": (40.8081, -73.9229),
    "bx-co-op-city": (40.8746, -73.8294),
    "bx-tremont": (40.8490, -73.8876),
    "si-st-george": (40.6445, -74.0768),
    "si-new-spring": (40.5889, -74.1650),
}


def _deterministic_jitter(
    nta_id: str, base_lat: float, base_lon: float
) -> tuple[float, float]:
    seed = sum(ord(c) for c in nta_id)
    rng = random.Random(seed)
    return (
        base_lat + rng.uniform(-0.006, 0.006),
        base_lon + rng.uniform(-0.008, 0.008),
    )


def render_map_view(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No data to display on map.")
        return

    st.subheader("Neighborhood Map")
    st.caption(
        "Use this map to see where your strongest matches cluster. Marker positions are approximate, not exact neighborhood boundaries."
    )

    zone_lookup = _load_nta_zone_lookup()

    # Build points dataframe
    rows = []
    for _, row in df.iterrows():
        nta_id = str(row.get("nta_id", ""))
        prefix = nta_id[:2].upper()
        zone_id = str(zone_lookup.get(nta_id, "")).strip().lower()

        # Prefer known neighborhood anchors; fallback to borough-level approximation.
        base_coords = ZONE_CENTERS.get(zone_id) or BOROUGH_CENTROIDS.get(prefix)
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
    points_df["marker_size"] = 7

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # If results span multiple boroughs (e.g., sidebar borough == Any),
        # let users quickly focus one borough in the map only.
        available_prefixes = sorted(
            p
            for p in points_df["nta_id"].astype(str).str[:2].str.upper().unique().tolist()
            if p in PREFIX_TO_BOROUGH
        )
        available_boroughs = [PREFIX_TO_BOROUGH[p] for p in available_prefixes]

        map_df = points_df
        if len(available_boroughs) > 1:
            chosen_borough = st.selectbox(
                "Choose borough",
                ["All boroughs"] + available_boroughs,
                help="Optional map-only focus when your current results include multiple boroughs.",
            )
            if chosen_borough != "All boroughs":
                chosen_prefix = next(
                    (k for k, v in PREFIX_TO_BOROUGH.items() if v == chosen_borough),
                    "",
                )
                if chosen_prefix:
                    map_df = points_df[
                        points_df["nta_id"].astype(str).str.startswith(chosen_prefix)
                    ].copy()

        if map_df.empty:
            st.info("No current results are shown for that borough.")
            return

        # "Current neighborhood" defaults to the top-ranked item in the current map scope.
        name_to_nta = dict(zip(map_df["label"], map_df["nta_id"]))
        neighborhood_names = sorted(name_to_nta.keys())
        current_choice = st.selectbox(
            "Current neighborhood",
            ["Auto (Top match)"] + neighborhood_names,
            help=(
                "Auto uses the highest-scoring neighborhood in this view. "
                "The current neighborhood is highlighted as a square block marker."
            ),
        )
        if current_choice == "Auto (Top match)":
            current_row = map_df.sort_values("final_score", ascending=False).iloc[0]
        else:
            current_row = map_df[map_df["label"] == current_choice].iloc[0]

        center_lat, center_lon = float(current_row["lat"]), float(current_row["lon"])
        zoom = 11

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="market_type",
            color_discrete_map=MARKET_TYPE_COLOR,
            size="marker_size",
            size_max=9,
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

        # Block-style highlight for the selected current neighborhood.
        fig.add_trace(
            go.Scattermapbox(
                lat=[current_row["lat"]],
                lon=[current_row["lon"]],
                mode="markers+text",
                text=["Current"],
                textposition="top center",
                marker=dict(size=16, color="#111111", symbol="square"),
                name="Current neighborhood",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "NTA: %{customdata[1]}<br>"
                    "Score: %{customdata[2]:.3f}<extra></extra>"
                ),
                customdata=[
                    [
                        str(current_row["label"]),
                        str(current_row["nta_id"]),
                        float(current_row["final_score"]),
                    ]
                ],
            )
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
