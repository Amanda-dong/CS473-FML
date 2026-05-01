from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    # Demand
    demand_prior: float = 10.0
    low_confidence_threshold: int = 30
    latent_implicit_weight: float = 0.40
    latent_keyword_weight: float = 0.35
    latent_activity_weight: float = 0.25
    # Gap
    gap_demand_blend: float = 0.60
    gap_latent_blend: float = 0.40
    # K-means
    kmeans_k: int = 4
    kmeans_max_iter: int = 300
    kmeans_tol: float = 1e-4
    kmeans_random_state: int = 42
    kmeans_confidence_epsilon: float = 1e-9
    # Phase 2 scoring
    score_demand_weight: float = 0.40
    score_gap_weight: float = 0.40
    score_viability_weight: float = 0.20
    # Phase 3 adjustments
    risk_penalty: float = 0.15
    forecast_boost: float = 0.10
    # GMM
    gmm_n_components: int = 4
    gmm_random_state: int = 42
    # Ridge
    ridge_alpha: float = 1.0
    ridge_cv_folds: int = 5
    ridge_random_state: int = 42
    forecast_min_reviews: int = 3
    # Similarity
    similarity_features: tuple[str, ...] = field(default_factory=lambda: ('demand_score', 'halal_supply_rate', 'gap_score', 'viability_score'))
    similarity_top_n: int = 3
    # Viability/Risk Thresholds
    viability_low_threshold: float = 0.6
    viability_medium_threshold: float = 0.35
    risk_high_threshold: float = 0.66
    risk_medium_threshold: float = 0.33
    # Domain
    halal_cuisines: frozenset[str] = field(default_factory=lambda: frozenset({'halal', 'middle eastern', 'pakistani', 'bangladeshi', 'afghan', 'egyptian', 'turkish', 'moroccan', 'lebanese', 'persian/iranian'}))
    halal_keywords: tuple[str, ...] = field(default_factory=lambda: ('halal', 'no pork', 'pork free', 'muslim', 'zabiha', 'zabihah', 'halal option', 'halal certified'))


CFG = ModelConfig()
