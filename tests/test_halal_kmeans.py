import numpy as np
import pandas as pd
import pytest
from src.halal_kmeans import HalalKMeans, run_kmeans
from src.config import CFG

def _df():
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({'nta_id':[f'NTA{i:03d}' for i in range(n)],'demand_score':np.concatenate([rng.uniform(0.7,1.0,10),rng.uniform(0.5,0.7,10),rng.uniform(0.2,0.4,10),rng.uniform(0.0,0.2,10)]),'latent_demand_score':np.concatenate([rng.uniform(0.7,1.0,10),rng.uniform(0.4,0.7,10),rng.uniform(0.2,0.5,10),rng.uniform(0.0,0.3,10)]),'halal_supply_rate':np.concatenate([rng.uniform(0.0,0.1,10),rng.uniform(0.1,0.3,10),rng.uniform(0.0,0.1,10),rng.uniform(0.0,0.05,10)]),'halal_cuisine_diversity_norm':rng.uniform(0,1,n)})

FEAT = ['demand_score','latent_demand_score','halal_supply_rate','halal_cuisine_diversity_norm']

def test_run_kmeans_shape():
    df = _df()
    r, km = run_kmeans(df, FEAT, k=4, cfg=CFG)
    assert len(r) == len(df)
    assert 'cluster_id' in r.columns and 'market_type' in r.columns

def test_confidence_range():
    r, _ = run_kmeans(_df(), FEAT, k=4, cfg=CFG)
    assert r['cluster_confidence'].between(0,1).all()
    assert r['cluster_confidence'].isna().sum() == 0

def test_market_types_valid():
    r, _ = run_kmeans(_df(), FEAT, k=4, cfg=CFG)
    valid = {'High Opportunity','Established Hub','Growing Market','Low Demand','Other'}
    assert set(r['market_type'].unique()).issubset(valid)

def test_centroid_shape_4_features():
    _, km = run_kmeans(_df(), FEAT, k=4, cfg=CFG)
    assert km.centroids_.shape[1] == 4
