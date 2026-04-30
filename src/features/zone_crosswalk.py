"""Zone-to-NTA crosswalk for bridging micro-zone IDs to NYC NTA boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Maps each micro-zone ID to real 2020 NYC NTA codes (6-char format: BK0202).
_BASE_ZONE_TO_NTA: dict[str, list[str]] = {
    # Brooklyn
    "bk-tandon": ["BK0202"],  # Downtown Brooklyn-DUMBO-Boerum Hill
    "bk-downtownbk": ["BK0202"],  # Downtown Brooklyn-DUMBO-Boerum Hill
    "bk-williamsburg": ["BK0102"],  # Williamsburg
    "bk-navy-yard": ["BK0202", "BK0261"],  # Downtown Brooklyn + Brooklyn Navy Yard
    "bk-fort-greene": ["BK0203"],  # Fort Greene
    "bk-crown-hts": ["BK0802"],  # Crown Heights (North)
    "bk-sunset-pk": ["BK0702", "BK0703"],  # Sunset Park (West + Central)
    # Manhattan
    "mn-midtown-e": ["MN0604"],  # East Midtown-Turtle Bay
    "mn-fidi": ["MN0101", "MN0102"],  # Financial District-Battery Park City + Tribeca
    "mn-columbia": ["MN0901"],  # Morningside Heights
    "mn-nyu-wash-sq": ["MN0202", "MN0201"],  # Greenwich Village + SoHo-Little Italy
    "mn-ues-hosp": ["MN0801"],  # Upper East Side-Lenox Hill-Roosevelt Island
    "mn-chelsea": ["MN0401"],  # Chelsea-Hudson Yards
    "mn-harlem": ["MN1001", "MN1002"],  # Harlem (South + North)
    "mn-lic-adj": ["MN0604", "MN0502"],  # East Midtown + Midtown-Times Square
    # Queens
    "qn-lic": ["QN0201"],  # Long Island City-Hunters Point
    "qn-astoria": ["QN0101", "QN0102", "QN0103"],  # Astoria (North + Old Astoria + Central)
    "qn-flushing": ["QN0707", "QN0704"],  # Flushing-Willets Point + Murray Hill-Broadway Flushing
    "qn-jackson-hts": ["QN0301"],  # Jackson Heights
    "qn-forest-hills": ["QN0602"],  # Forest Hills
    "qn-jamaica": ["QN1201"],  # Jamaica
    # Queens expansion zones
    "qn-college-point-whitestone": ["QN0701", "QN0702"],  # College Point + Whitestone-Beechhurst
    "qn-murray-hill-flushing": ["QN0704"],  # Murray Hill-Broadway Flushing
    "qn-elmhurst-corona": ["QN0401", "QN0402"],  # Elmhurst + Corona
    "qn-rego-middle": ["QN0601", "QN0504"],  # Rego Park + Middle Village
    # Bronx
    "bx-fordham": ["BX0701", "BX0503"],  # University Heights-Fordham + Fordham Heights
    "bx-mott-haven": ["BX0101"],  # Mott Haven-Port Morris
    "bx-co-op-city": ["BX1004"],  # Co-op City
    "bx-tremont": ["BX0602"],  # Tremont
    # Bronx expansion zones
    "bx-south-hub": ["BX0301", "BX0102", "BX0201"],  # Morrisania + Melrose + Hunts Point
    "bx-west-corridor": ["BX0801", "BX0803"],  # Kingsbridge Heights + Riverdale
    "bx-east-corridor": ["BX0904", "BX1002"],  # Parkchester + Throgs Neck-Schuylerville
    # Staten Island
    "si-st-george": ["SI0101"],  # St. George-New Brighton
    "si-new-spring": ["SI0204"],  # New Springville-Willowbrook-Bulls Head-Travis
    # Brooklyn expansion zones
    "bk-bushwick-ridgewood": ["BK0401", "BK0402"],  # Bushwick (West + East)
    "bk-bayridge-benson": ["BK1001", "BK1101"],  # Bay Ridge + Bensonhurst
    "bk-flatbush-midwood": ["BK1401", "BK1403"],  # Flatbush + Midwood
}


def _generic_zone_id(nta_code: str) -> str:
    """Create a stable synthetic zone id for full-NTA fallback coverage."""
    return f"nta-{nta_code.strip().lower()}"


def _load_all_nta_codes() -> list[str]:
    """Load all NYC 2020 NTA codes from local GeoJSON when available."""
    for candidate in ("data/raw/nta.geojson", "data/raw/nta2020_nyc.geojson"):
        geojson_path = Path(candidate)
        if not geojson_path.exists():
            continue
        try:
            payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        features = payload.get("features", [])
        codes: set[str] = set()
        for feature in features:
            props = feature.get("properties", {}) if isinstance(feature, dict) else {}
            nta = str(props.get("nta2020", "")).strip()
            if nta:
                codes.add(nta)
        return sorted(codes)
    return []


_NTA_2010_TO_2020: dict[str, str] = {
    "BK09": "BK0202",
    "BK31": "BK1001",
    "BK32": "BK1001",
    "BK33": "BK0203",
    "BK41": "BK1401",
    "BK42": "BK0702",
    "BK43": "BK1401",
    "BK69": "BK0802",
    "BK73": "BK0102",
    "BK77": "BK0401",
    "BX01": "BX0101",
    "BX03": "BX0301",
    "BX05": "BX0301",
    "BX06": "BX0701",
    "BX07": "BX0801",
    "BX08": "BX0801",
    "BX09": "BX0602",
    "BX26": "BX0904",
    "BX44": "BX1004",
    "BX46": "BX0904",
    "MN09": "MN0901",
    "MN11": "MN1001",
    "MN17": "MN0604",
    "MN19": "MN0604",
    "MN21": "MN0401",
    "MN22": "MN0202",
    "MN25": "MN0101",
    "MN31": "MN0801",
    "QN17": "QN0602",
    "QN18": "QN0601",
    "QN26": "QN0401",
    "QN27": "QN0401",
    "QN48": "QN0707",
    "QN49": "QN0701",
    "QN50": "QN0704",
    "QN57": "QN0301",
    "QN61": "QN1201",
    "QN70": "QN0201",
    "QN72": "QN0101",
    "SI07": "SI0101",
    "SI11": "SI0204",
}


def _build_zone_to_nta() -> dict[str, list[str]]:
    """Return crosswalk that preserves base zones and fills all remaining NTAs."""
    mapping = {zone_id: list(ntas) for zone_id, ntas in _BASE_ZONE_TO_NTA.items()}
    covered = {nta for ntas in mapping.values() for nta in ntas}
    for nta in _load_all_nta_codes():
        if nta in covered:
            continue
        mapping[_generic_zone_id(nta)] = [nta]
    return mapping


ZONE_TO_NTA: dict[str, list[str]] = _build_zone_to_nta()

# Reverse lookup: NTA code -> list of zone IDs
NTA_TO_ZONES: dict[str, list[str]] = {}
for _zone, _ntas in ZONE_TO_NTA.items():
    for _nta in _ntas:
        NTA_TO_ZONES.setdefault(_nta, []).append(_zone)

# When one NTA maps to multiple micro-zones, pick a single primary for point-level assignment.
# (Aggregations that split one NTA across zones still use aggregate_nta_to_zone.)
NTA_PRIMARY_ZONE: dict[str, str] = {
    "BK0202": "bk-downtownbk",  # shared by bk-tandon, bk-downtownbk, bk-navy-yard
    "MN0604": "mn-midtown-e",  # shared by mn-midtown-e, mn-lic-adj
    "QN0704": "qn-flushing",  # shared by qn-flushing, qn-murray-hill-flushing
}


def resolve_nta_to_zone_id(nta: str | None) -> str | None:
    """Map an ACS NTA code (e.g. ``MN0202``) to one micro-zone ``zone_id``.

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

    # Backward-compat: also map 4-char prefixes of 6-char codes so ETL data
    # using 2010 NTA codes (BK02) can join against 2020 codes (BK0202 -> prefix BK02).
    # We deduplicate so one 4-char prefix maps to the first zone found.
    seen_4char: set[str] = set()
    extra_rows = []
    for zone_id, nta_list in ZONE_TO_NTA.items():
        for nta in nta_list:
            if len(nta) == 6:
                prefix = nta[:4]
                if prefix not in seen_4char:
                    seen_4char.add(prefix)
                    extra_rows.append({"zone_id": zone_id, zone_col: prefix})
    if extra_rows:
        mapping = pd.concat([mapping, pd.DataFrame(extra_rows)], ignore_index=True)

    extra_2010 = []
    for nta_2010, nta_2020 in _NTA_2010_TO_2020.items():
        for zone_id, nta_list in ZONE_TO_NTA.items():
            if nta_2020 in nta_list:
                extra_2010.append({"zone_id": zone_id, zone_col: nta_2010})
                break
    if extra_2010:
        mapping = pd.concat([mapping, pd.DataFrame(extra_2010)], ignore_index=True)
        mapping = mapping.drop_duplicates(subset=[zone_col, "zone_id"])

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
