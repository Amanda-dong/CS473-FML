import pandas as pd
import pytest
from src.config import CFG

def _reviews():
    return pd.DataFrame({'review_id':[f'r{i}' for i in range(10)],'restaurant_id':[f'rid{i}' for i in range(10)],'nta':['MN22']*5+['BK10']*5,'review_date':['2023-01-01']*10,'review_text':['this place is halal certified','no pork anywhere love it','amazing pizza','zabiha beef perfect','great lunch','halal option available','nothing special','pork free delicious','ordinary restaurant','muslim friendly'],'rating':[4.0]*10})

def _labels():
    return pd.DataFrame({'review_id':[f'r{i}' for i in range(10)],'halal_relevance':['explicit_halal','implicit_halal','not_related','implicit_halal','not_related','explicit_halal','not_related','implicit_halal','not_related','implicit_halal']})

def test_latent_demand_columns():
    from src.halal_demand import build_latent_demand
    r = build_latent_demand(_reviews(), _labels(), 'review_id', 'halal_relevance', CFG)
    for c in ['nta_id','latent_demand_score','activity_score']:
        assert c in r.columns

def test_latent_demand_range():
    from src.halal_demand import build_latent_demand
    r = build_latent_demand(_reviews(), _labels(), 'review_id', 'halal_relevance', CFG)
    assert r['latent_demand_score'].between(0,1).all()
    assert r['activity_score'].between(0,1).all()

def test_mn22_latent_gt_revealed():
    from src.halal_demand import build_demand
    df = build_demand()
    mn22 = df[df['nta_id']=='MN22'].iloc[0]
    assert mn22['latent_demand_score'] > mn22['demand_score'], f'latent {mn22["latent_demand_score"]:.3f} should exceed demand {mn22["demand_score"]:.3f}'
