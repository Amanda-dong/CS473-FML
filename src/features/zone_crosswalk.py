"""Zone-to-NTA crosswalk for bridging micro-zone IDs to NYC NTA boundaries."""

from __future__ import annotations

import pandas as pd

# Maps each of the 27 micro-zone IDs to 2020 NYC NTA codes.
ZONE_TO_NTA: dict[str, list[str]] = {
    # Brooklyn
    "bk-tandon": ["BK09"],  # Downtown Brooklyn-DUMBO-Boerum Hill
    "bk-downtownbk": ["BK09"],  # Downtown Brooklyn-DUMBO-Boerum Hill
    "bk-williamsburg": ["BK73"],  # Williamsburg
    "bk-navy-yard": ["BK09", "BK33"],  # Downtown BK + Vinegar Hill/Fort Greene adj.
    "bk-fort-greene": ["BK33"],  # Fort Greene
    "bk-crown-hts": ["BK69"],  # Crown Heights North
    "bk-sunset-pk": ["BK42"],  # Sunset Park
    # Manhattan
    "mn-midtown-e": ["MN17"],  # Midtown-Midtown South
    "mn-fidi": ["MN25"],  # Battery Park City-Lower Manhattan
    "mn-columbia": ["MN09"],  # Morningside Heights
    "mn-nyu-wash-sq": ["MN22"],  # Greenwich Village-SoHo
    "mn-ues-hosp": ["MN31"],  # Lenox Hill-Roosevelt Island
    "mn-chelsea": ["MN21"],  # Chelsea-Hudson Yards
    "mn-harlem": ["MN11"],  # Central Harlem
    "mn-lic-adj": ["MN17", "MN19"],  # Midtown East + Turtle Bay
    # Queens
    "qn-lic": ["QN70"],  # Long Island City
    "qn-astoria": ["QN72"],  # Astoria
    "qn-flushing": ["QN48"],  # Flushing
    "qn-jackson-hts": ["QN57"],  # Jackson Heights
    "qn-forest-hills": ["QN17"],  # Forest Hills
    "qn-jamaica": ["QN61"],  # Jamaica
    # Bronx
    "bx-fordham": ["BX06"],  # Fordham
    "bx-mott-haven": ["BX01"],  # Mott Haven-Port Morris
    "bx-co-op-city": ["BX44"],  # Co-op City
    "bx-tremont": ["BX09"],  # East Tremont
    # Staten Island
    "si-st-george": ["SI07"],  # St. George
    "si-new-spring": ["SI11"],  # New Springville-Bloomfield-Travis
}

# Reverse lookup: NTA code -> list of zone IDs
NTA_TO_ZONES: dict[str, list[str]] = {}
for _zone, _ntas in ZONE_TO_NTA.items():
    for _nta in _ntas:
        NTA_TO_ZONES.setdefault(_nta, []).append(_zone)

# When one ACS NTA code maps to multiple micro-zones, pick a single primary for point-level assignment.
# (Aggregations that split one NTA across zones still use aggregate_nta_to_zone.)
NTA_PRIMARY_ZONE: dict[str, str] = {
    "BK09": "bk-tandon",
    "BK33": "bk-fort-greene",
    "MN17": "mn-midtown-e",
}


def resolve_nta_to_zone_id(nta: str | None) -> str | None:
    """Map an ACS NTA code (e.g. ``MN22``) to one micro-zone ``zone_id``.

    Returns ``None`` if the NTA is not part of :data:`ZONE_TO_NTA` (e.g. a
    NYC block outside the modeled micro-zone list).
    """
    if nta is None or (isinstance(nta, float) and pd.isna(nta)):
        return None
    code = str(nta).strip().upper()
    if not code:
        return None
    zones = NTA_TO_ZONES.get(code)
    if not zones:
        return None
    if len(zones) == 1:
        return zones[0]
    return NTA_PRIMARY_ZONE.get(code, sorted(zones)[0])


def aggregate_nta_to_zone(
    nta_df: pd.DataFrame,
    zone_col: str = "nta_id",
    agg_rules: dict[str, str] | None = None,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate NTA-level data to zone-level using the crosswalk.

    Parameters
    ----------
    nta_df:
        DataFrame with an NTA identifier column and optionally a year/time column.
    zone_col:
        Name of the column containing NTA codes.
    agg_rules:
        Mapping of column name -> aggregation function (e.g. {"population": "sum",
        "median_income": "mean"}). Numeric columns not listed default to "mean".
    weights_col:
        Optional column for population/sample-weighted aggregation. When
        provided, numeric columns use weighted mean instead of simple mean
        (unless overridden by ``agg_rules``).

    Returns
    -------
    DataFrame with ``zone_id`` replacing the NTA column, aggregated per zone (and
    per ``year``/``time_key`` if present).
    """
    if nta_df.empty or zone_col not in nta_df.columns:
        return pd.DataFrame()

    # Build exploded mapping frame
    rows = []
    for zone_id, nta_list in ZONE_TO_NTA.items():
        for nta in nta_list:
            rows.append({"zone_id": zone_id, zone_col: nta})
    mapping = pd.DataFrame(rows)

    merged = nta_df.merge(mapping, on=zone_col, how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Determine groupby keys
    time_col = (
        "year"
        if "year" in merged.columns
        else ("time_key" if "time_key" in merged.columns else None)
    )
    group_keys = ["zone_id"] + ([time_col] if time_col else [])

    # Build aggregation dict
    numeric_cols = merged.select_dtypes(include=["number"]).columns.tolist()
    skip_cols = set(group_keys) | ({weights_col} if weights_col else set())
    default_agg = {c: "mean" for c in numeric_cols if c not in skip_cols}
    if agg_rules:
        default_agg.update(agg_rules)

    if not default_agg:
        return merged[group_keys].drop_duplicates().reset_index(drop=True)

    # Weighted aggregation when weights_col is available
    if weights_col and weights_col in merged.columns:
        import numpy as _np

        def _weighted_agg(grp: pd.DataFrame) -> pd.Series:
            w = grp[weights_col].values.astype(float)
            w_sum = w.sum()
            out = {}
            for col, func in default_agg.items():
                if col not in grp.columns:
                    continue
                if func == "mean" and w_sum > 0:
                    out[col] = float(_np.dot(w, grp[col].values.astype(float)) / w_sum)
                elif func == "sum":
                    out[col] = float(grp[col].sum())
                else:
                    out[col] = float(grp[col].agg(func))
            return pd.Series(out)

        result = (
            merged.groupby(group_keys)
            .apply(_weighted_agg, include_groups=False)
            .reset_index()
        )
    else:
        result = merged.groupby(group_keys, as_index=False).agg(default_agg)
    if time_col and time_col != "time_key":
        result = result.rename(columns={time_col: "time_key"})
    return result
