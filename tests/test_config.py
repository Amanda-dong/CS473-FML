from src.config import CFG, ModelConfig
import pytest

def test_defaults_sane():
    assert abs(CFG.score_demand_weight + CFG.score_gap_weight + CFG.score_viability_weight - 1.0) < 1e-9
    assert CFG.kmeans_k == 4
    assert 'halal' in CFG.halal_cuisines
    assert len(CFG.halal_keywords) >= 5

def test_immutable():
    with pytest.raises(Exception):
        CFG.kmeans_k = 99

def test_custom_override():
    c = ModelConfig(kmeans_k=3)
    assert c.kmeans_k == 3
    assert c.demand_prior == CFG.demand_prior
