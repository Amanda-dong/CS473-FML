import pandas as pd
import pytest
from src.config import CFG

def _supply():
    return pd.DataFrame({'nta_id':['MN22','BK10','QN25'],'total_restaurants':[50,100,30],'halal_restaurants':[2,5,0],'halal_supply_rate':[0.04,0.05,0.0],'halal_cuisine_diversity':[1.0,2.0,0.0]})

def _demand():
    return pd.DataFrame({'nta_id':['MN22','BK10','QN25'],'demand_score':[0.33,0.80,0.60],'latent_demand_score':[0.53,0.85,0.55],'total_reviews':[423,800,300],'halal_related_share':[0.33,0.55,0.25],'explicit_halal_share':[0.22,0.40,0.10],'shrunk_share':[0.33,0.55,0.25],'review_count_flag':['high confidence']*3})

def test_build_gap_columns():
    from src.halal_opportunity import build_gap
    r = build_gap(_demand(), _supply(), CFG)
    for c in ['gap_score','supply_norm','combined_demand','halal_cuisine_diversity_norm']:
        assert c in r.columns

def test_build_gap_range():
    from src.halal_opportunity import build_gap
    r = build_gap(_demand(), _supply(), CFG)
    assert r['gap_score'].between(0,1).all()
    assert r['supply_norm'].between(0,1).all()
    assert r['halal_cuisine_diversity_norm'].between(0,1).all()

def test_build_supply_no_hygiene():
    from pathlib import Path
    if Path('data/raw/restaurant_hygiene.csv').exists():
        pytest.skip('hygiene.csv present')
    from src.halal_opportunity import build_supply
    df = build_supply()
    assert len(df) > 0

def test_build_supply_no_nan():
    from src.halal_opportunity import build_supply
    df = build_supply()
    assert df['halal_restaurants'].isna().sum() == 0
    assert df['total_restaurants'].isna().sum() == 0
