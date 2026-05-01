from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Optional dependencies for spatial analysis
try:
    from shapely.wkt import loads as wkt_loads
    import libpysal
    from esda.moran import Moran_Local
    HAS_SPATIAL_LIBS = True
except ImportError:
    HAS_SPATIAL_LIBS = False

ROOT = Path(__file__).resolve().parents[1]
NTA_BOUNDARIES = ROOT / 'data' / 'raw' / 'nta_boundaries.csv'

def _load_centroids() -> pd.DataFrame:
    """
    Parse NTA centroids from NTA boundaries CSV.
    
    Parses WKT MULTIPOLYGON strings, computes centroids, and returns
    a DataFrame with nta_id, lon, and lat.
    """
    if not NTA_BOUNDARIES.exists():
        return pd.DataFrame(columns=['nta_id', 'lon', 'lat'])
    
    try:
        df = pd.read_csv(NTA_BOUNDARIES)
        if 'the_geom' not in df.columns or 'NTA2020' not in df.columns:
            return pd.DataFrame(columns=['nta_id', 'lon', 'lat'])
        
        def _parse_centroid(row):
            try:
                geom = wkt_loads(row['the_geom'])
                c = geom.centroid
                return pd.Series({'nta_id': row['NTA2020'], 'lon': c.x, 'lat': c.y})
            except Exception:
                return pd.Series({'nta_id': row['NTA2020'], 'lon': float('nan'), 'lat': float('nan')})
        centroid_df = df.apply(_parse_centroid, axis=1).dropna(subset=['lon', 'lat'])
        return centroid_df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['nta_id', 'lon', 'lat'])

def build_lisa(gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Local Moran's I (LISA) for gap_score.
    
    Args:
        gap_df: DataFrame with columns 'nta_id' and 'gap_score'
        
    Returns:
        DataFrame with columns: 
            nta_id, moran_ii, moran_p, moran_q, lisa_opportunity
        
        lisa_opportunity is True if moran_q == 2 (Low-High quadrant),
        meaning a low gap score NTA surrounded by high gap score neighbors.
    """
    output_cols = ['nta_id', 'moran_ii', 'moran_p', 'moran_q', 'lisa_opportunity']
    
    # Graceful fallback if libs missing
    if not HAS_SPATIAL_LIBS:
        result = gap_df[['nta_id']].copy()
        for col in output_cols[1:]:
            result[col] = np.nan
        return result
    
    centroids_df = _load_centroids()
    
    # Join centroids with gap_df
    merged = gap_df.merge(centroids_df, on='nta_id', how='left')
    
    # Filter out missing geometries or missing scores for the calculation
    valid_mask = merged['lon'].notna() & merged['lat'].notna() & merged['gap_score'].notna()
    valid_data = merged[valid_mask].copy()
    
    # Need at least k+1 points to build weights
    if len(valid_data) <= 5:
        result = gap_df[['nta_id']].copy()
        for col in output_cols[1:]:
            result[col] = np.nan
        return result
    
    coords = valid_data[['lon', 'lat']].values
    values = valid_data['gap_score'].values
    
    try:
        # Build spatial weights using 5 nearest neighbors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = libpysal.weights.KNN.from_array(coords, k=5)
            w.transform = 'R' # Row-standardized
            
        # Compute Local Moran's I
        lisa = Moran_Local(values, w, permutations=999)
        
        valid_data['moran_ii'] = lisa.Is
        valid_data['moran_p'] = lisa.p_sim
        valid_data['moran_q'] = lisa.q
        
        # Quadrants: 'HH'=1, 'LH'=2, 'LL'=3, 'HL'=4
        # lisa_opportunity: True if moran_q == 2 (Low-High)
        valid_data['lisa_opportunity'] = (valid_data['moran_q'] == 2)
        
        # Merge back to include all original NTAs from gap_df
        result = gap_df[['nta_id']].merge(
            valid_data[['nta_id', 'moran_ii', 'moran_p', 'moran_q', 'lisa_opportunity']],
            on='nta_id',
            how='left'
        )
        # Ensure boolean type for lisa_opportunity, filling NaNs with False
        result['lisa_opportunity'] = result['lisa_opportunity'].fillna(False).astype(bool)
        
        return result
        
    except Exception:
        # Fallback if calculation fails
        result = gap_df[['nta_id']].copy()
        for col in output_cols[1:]:
            result[col] = np.nan
        return result
