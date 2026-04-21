"""Demo page: Yelp reviews × micro-zones with a map.

Run from repo root::

    streamlit run frontend/app.py

Open **Review Zones Demo** in the sidebar. Requires:

- ``data/processed/yelp_reviews_with_zones.csv`` (from ``scripts/join_reviews_to_zones.py``)
- ``data/raw/yelp_business.csv`` (for coordinates on the map)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = REPO_ROOT / "data" / "processed" / "yelp_reviews_with_zones.csv"
BUSINESS_PATH = REPO_ROOT / "data" / "raw" / "yelp_business.csv"

_MAP_KEY = "reviews_zone_map_plotly"


def _restaurant_id_from_plotly_state(
    plotly_state, fig, map_plot: pd.DataFrame
) -> str | None:
    """Read selected point from st.plotly_chart(on_select=...) return value."""
    if plotly_state is None:
        return None
    sel = getattr(plotly_state, "selection", None)
    if sel is None and isinstance(plotly_state, dict):
        sel = plotly_state.get("selection")
    if sel is None:
        return None
    pts = getattr(sel, "points", None)
    if pts is None and isinstance(sel, dict):
        pts = sel.get("points")
    if not pts:
        return None
    p0 = pts[0]
    if not isinstance(p0, dict):
        return None
    cd = p0.get("customdata")
    if hasattr(cd, "tolist"):
        cd = cd.tolist()
    if isinstance(cd, (list, tuple)) and len(cd) > 0:
        return str(cd[0])
    if isinstance(cd, str) and cd:
        return cd
    # Fallback: pointIndex + curveNumber into fig.data[].customdata
    idx = p0.get("pointIndex")
    cnum = p0.get("curveNumber", 0)
    if (
        idx is not None
        and fig is not None
        and hasattr(fig, "data")
        and cnum < len(fig.data)
    ):
        trace = fig.data[cnum]
        raw = getattr(trace, "customdata", None)
        if raw is not None and len(raw) > idx:
            row = raw[idx]
            return str(row[0]) if isinstance(row, (list, tuple)) else str(row)
    # Last resort: match lat/lon from selection
    lat, lon = p0.get("lat"), p0.get("lon")
    if lat is not None and lon is not None:
        m = map_plot[
            ((map_plot["latitude"] - float(lat)).abs() < 1e-4)
            & ((map_plot["longitude"] - float(lon)).abs() < 1e-4)
        ]
        if len(m) == 1:
            return str(m["restaurant_id"].iloc[0])
    return None


st.set_page_config(page_title="Review × Zones Demo", layout="wide")

st.title("Reviews × Micro-zones")
st.caption(
    "`zone_id` is filled only when the business falls in a modeled micro-zone "
    "(see `zone_crosswalk.py`); `nta` may still be set for any NYC NTA."
)

if not REVIEWS_PATH.is_file():
    st.error(f"Missing dataset: `{REVIEWS_PATH}`")
    st.info("Run from the repo root: `python scripts/join_reviews_to_zones.py`")
    st.stop()

df = pd.read_csv(REVIEWS_PATH)
df["has_zone"] = df["zone_id"].notna() & (df["zone_id"].astype(str).str.strip() != "")

biz_full: pd.DataFrame | None = None
if BUSINESS_PATH.is_file():
    biz_full = pd.read_csv(BUSINESS_PATH)
    biz_full["id"] = biz_full["id"].astype(str).str.strip()
    biz_geo = biz_full[["id", "latitude", "longitude"]].rename(
        columns={"id": "restaurant_id"}
    )
    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    df = df.merge(biz_geo, on="restaurant_id", how="left")
else:
    st.warning(f"No `{BUSINESS_PATH}` — map disabled.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Review rows", f"{len(df):,}")
c2.metric("With zone_id", f"{int(df['has_zone'].sum()):,}")
c3.metric("Share with zone_id", f"{100 * df['has_zone'].mean():.1f}%")
if "in_nyc_nta" in df.columns:
    im = df["in_nyc_nta"].fillna(False).astype(bool)
    c4.metric("in_nyc_nta", f"{int(im.sum()):,}")

st.divider()

# ---- Map: one small dot per restaurant (size fixed; review count in hover only) ----
if {"latitude", "longitude"}.issubset(df.columns):
    map_agg = (
        df.dropna(subset=["latitude", "longitude"])
        .groupby("restaurant_id", as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            zone_id=("zone_id", "first"),
            n_reviews=("review_id", "count"),
        )
    )
    map_agg["zone_label"] = map_agg["zone_id"].fillna("(no zone)").astype(str)

    # Keep NYC metro so the map does not zoom out to meaningless extents
    _in_bbox = map_agg["latitude"].between(40.49, 40.92) & map_agg["longitude"].between(
        -74.05, -73.70
    )
    map_plot = map_agg.loc[_in_bbox].copy()
    if map_plot.empty:
        map_plot = map_agg

    if biz_full is not None:
        map_plot = map_plot.merge(
            biz_full[["id", "name"]].rename(columns={"id": "restaurant_id"}),
            on="restaurant_id",
            how="left",
        )
        map_plot["name"] = map_plot["name"].fillna("").astype(str)
        map_plot["hover_title"] = map_plot.apply(
            lambda r: (str(r["name"]).strip() or str(r["restaurant_id"])),
            axis=1,
        )
    else:
        map_plot["name"] = map_plot["restaurant_id"].astype(str)
        map_plot["hover_title"] = map_plot["restaurant_id"].astype(str)

    st.subheader("Map (restaurant locations)")
    st.caption(
        "**Hover** a dot to see the **restaurant name**. **Click** to load reviews below. "
        "Colors = micro-zone (`zone_id`)."
    )

    try:
        import plotly.express as px

        center_lat = float(map_plot["latitude"].median())
        center_lon = float(map_plot["longitude"].median())
        fig = px.scatter_mapbox(
            map_plot,
            lat="latitude",
            lon="longitude",
            color="zone_label",
            custom_data=["restaurant_id"],
            hover_name="hover_title",
            hover_data={
                "restaurant_id": True,
                "name": True,
                "latitude": ":.5f",
                "longitude": ":.5f",
                "n_reviews": True,
                "zone_label": True,
            },
            zoom=11.4,
            height=560,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # scattermapbox markers do not support marker.line (unlike scattergeo)
        fig.update_traces(marker=dict(size=9, opacity=0.88))
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon)),
            legend_title_text="zone_id",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.85)",
            ),
            showlegend=True,
        )
        plotly_state = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=_MAP_KEY,
        )
        rid_map = _restaurant_id_from_plotly_state(plotly_state, fig, map_plot)
        if rid_map is None:
            rid_map = _restaurant_id_from_plotly_state(
                st.session_state.get(_MAP_KEY), fig, map_plot
            )
        if rid_map:
            st.session_state["active_restaurant_id"] = str(rid_map)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Plotly map failed ({exc}). Falling back to `st.map`.")
        slim = map_agg.rename(columns={"longitude": "lon", "latitude": "lat"})[
            ["lat", "lon"]
        ].drop_duplicates()
        st.map(slim)

    # ---- One restaurant: search name, map click, or dropdown ----
    st.subheader("Reviews for one restaurant")
    ids_ordered = (
        map_plot.sort_values("n_reviews", ascending=False)["restaurant_id"]
        .drop_duplicates()
        .tolist()
    )
    name_by_id = (
        map_plot.drop_duplicates(subset=["restaurant_id"])
        .set_index("restaurant_id")["name"]
        .fillna("")
        .astype(str)
    )

    search_q = st.text_input(
        "Search restaurant name",
        key="restaurant_name_search",
        placeholder="Type any part of the name (case-insensitive)",
    )
    ql = search_q.strip().lower()
    if ql:
        ids_filtered = [
            rid
            for rid in ids_ordered
            if ql in name_by_id.get(rid, "").lower() or ql in str(rid).lower()
        ]
        if not ids_filtered:
            st.warning(
                "No restaurant names match that search. Clear the box to see all."
            )
    else:
        ids_filtered = list(ids_ordered)

    if not ids_ordered:
        st.caption("No restaurants on the map.")
    elif not ids_filtered:
        chosen = None
    else:
        if "active_restaurant_id" not in st.session_state:
            st.session_state["active_restaurant_id"] = ids_filtered[0]
        aid = st.session_state.get("active_restaurant_id")
        if aid not in ids_filtered:
            aid = ids_filtered[0]
            st.session_state["active_restaurant_id"] = aid
        idx = ids_filtered.index(str(aid))

        def _fmt_restaurant(rid: str) -> str:
            r = map_plot.loc[map_plot["restaurant_id"] == rid].iloc[0]
            nm = str(r.get("name", "") or "").strip() or "(no name)"
            return f"{nm} · {int(r['n_reviews'])} rev · {r['zone_label']}"

        chosen = st.selectbox(
            "Restaurant (map click, search above, or pick here)",
            options=ids_filtered,
            index=idx,
            format_func=_fmt_restaurant,
        )
        st.session_state["active_restaurant_id"] = str(chosen)

        sub = df[df["restaurant_id"] == chosen].copy()
        sort_cols = [c for c in ("time_key", "review_date") if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        # ---- Business profile (Yelp business CSV) ----
        if biz_full is not None:
            bmatch = biz_full[biz_full["id"] == chosen]
            if not bmatch.empty:
                b = bmatch.iloc[0]
                st.markdown(f"### {b.get('name', 'Unknown')}")
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric(
                    "Yelp rating",
                    f"{float(b['rating']):.1f}" if pd.notna(b.get("rating")) else "—",
                )
                b2.metric(
                    "Yelp review count",
                    int(b["review_count"]) if pd.notna(b.get("review_count")) else "—",
                )
                pr = b.get("price")
                b3.metric("Price", str(pr) if pd.notna(pr) and str(pr).strip() else "—")
                closed = bool(b.get("is_closed", False))
                b4.metric("Status", "Closed" if closed else "Open")
                n_fusion = int(
                    map_plot.loc[map_plot["restaurant_id"] == chosen, "n_reviews"].iloc[
                        0
                    ]
                )
                b5.metric("Rows in Fusion CSV", n_fusion)
                st.write(
                    f"**Categories:** {b.get('categories', '—')}  \n"
                    f"**Coordinates:** `{b.get('latitude', '—')}`, `{b.get('longitude', '—')}`  \n"
                    f"**Search term (pull):** `{b.get('search_term', '—')}` · "
                    f"**Anchor:** `{b.get('anchor_name', '—')}`"
                )
                if len(sub):
                    z0 = sub.iloc[0]
                    st.write(
                        f"**Micro-zone (`zone_id`):** `{z0.get('zone_id', '—')}` · "
                        f"**NTA:** `{z0.get('nta', '—')}`"
                    )

        st.subheader("Reviews in this dataset")
        st.metric("Review rows (Fusion export)", len(sub))
        rcols = [
            c for c in ("time_key", "rating", "zone_id", "nta") if c in sub.columns
        ]
        st.dataframe(
            sub[rcols + ["review_text"]],
            use_container_width=True,
            height=min(640, 140 + 36 * min(len(sub), 18)),
        )
else:
    st.info(
        "Add lat/lon by placing `yelp_business.csv` under `data/raw/` to enable the map."
    )

st.divider()

zc = df.loc[df["has_zone"], "zone_id"].value_counts().head(25)
if not zc.empty:
    st.subheader("Reviews with a zone (Top 25 zones)")
    st.bar_chart(zc)
else:
    st.warning("No rows with `zone_id` — chart skipped.")

st.subheader("Table preview")
zones = ["(all)"]
if "zone_id" in df.columns:
    zones.extend(sorted(df["zone_id"].dropna().astype(str).unique().tolist()))

sel = st.selectbox("Filter by zone_id", zones)
show = df if sel == "(all)" else df[df["zone_id"].astype(str) == sel]
n_show = st.slider("Rows to show", 5, 200, 20, 5)
cols = [
    c
    for c in (
        "review_id",
        "restaurant_id",
        "time_key",
        "zone_id",
        "nta",
        "rating",
        "latitude",
        "longitude",
    )
    if c in show.columns
]
st.dataframe(
    show[cols + ["review_text"]].head(n_show),
    use_container_width=True,
    height=420,
)

with st.expander("Column notes"):
    st.markdown(
        """
- **Empty `zone_id`**: business outside NYC NTAs, or inside an NTA that is not mapped to any
  micro-zone in `ZONE_TO_NTA`.
- **`time_key`**: calendar year from `review_date`.
- **`review_id`**: stable SHA-256 id for downstream Gemini labeling.
"""
    )
