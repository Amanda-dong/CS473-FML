import pytest
from pathlib import Path
FINAL = Path('data/output/final_recommendations.csv')

def test_final_no_nan_columns():
    if not FINAL.exists(): pytest.skip('output not yet generated')
    import pandas as pd
    df = pd.read_csv(FINAL)
    bad = [c for c in df.columns if df[c].isna().all()]
    assert bad == [], f'All-NaN columns: {bad}'

def test_final_has_latent_demand():
    if not FINAL.exists(): pytest.skip('output not yet generated')
    import pandas as pd
    df = pd.read_csv(FINAL)
    assert 'latent_demand_score' in df.columns
    assert 'cluster_confidence' in df.columns

def test_mn22_not_low_demand():
    if not FINAL.exists(): pytest.skip('output not yet generated')
    import pandas as pd
    df = pd.read_csv(FINAL)
    mn22 = df[df['nta_id']=='MN22']
    if mn22.empty: pytest.skip('MN22 not in output')
    assert mn22.iloc[0]['market_type'] != 'Low Demand', f'MN22 still Low Demand: {mn22.iloc[0]["market_type"]}'
