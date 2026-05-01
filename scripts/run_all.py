from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_phase1 as phase1
import scripts.run_phase2 as phase2
import scripts.run_phase3 as phase3

OUT = ROOT / 'data' / 'output'

PHASE1_COLS = ['nta_id','cluster_id','market_type','cluster_confidence','demand_score','latent_demand_score','halal_supply_rate','gap_score','total_restaurants','halal_restaurants']
PHASE2_COLS = ['nta_id','final_score','demand_score','gap_score','viability_score','similar_ntas']
FINAL_COLS  = ['nta_id','final_score_adjusted','market_type','high_risk_prob','risk_bucket','halal_demand_forecast']


def _validate(path, expected, label):
    df = pd.read_csv(path)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f'[{label}] missing columns: {missing}')
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        raise RuntimeError(f'[{label}] all-NaN columns: {all_nan}')
    if len(df) == 0:
        raise RuntimeError(f'[{label}] 0 rows')
    print(f'[{label}] OK — {len(df)} NTAs, {len(df.columns)} cols')


def _summary(path):
    df = pd.read_csv(path)
    top5 = df.nlargest(5, 'final_score_adjusted')[['nta_id','market_type','final_score_adjusted','risk_bucket']]
    print('\nTop 5 NTAs:')
    print(top5.to_string(index=False))
    print('\nMarket type distribution:')
    print(df['market_type'].value_counts().to_string())


def main():
    t0 = time.perf_counter()
    print('=== Phase 1 ===')
    phase1.main()
    _validate(OUT / 'phase1_cluster_assignments.csv', PHASE1_COLS, 'Phase1')
    print('\n=== Phase 2 ===')
    phase2.main()
    _validate(OUT / 'phase2_opportunity_scores.csv', PHASE2_COLS, 'Phase2')
    phase3.main()
    _validate(OUT / 'final_recommendations.csv', FINAL_COLS, 'Phase3')
    _summary(OUT / 'final_recommendations.csv')
    print(f'\nTotal: {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
