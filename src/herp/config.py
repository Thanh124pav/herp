"""HERP v3 configuration. Legacy heuristic allocation is intentionally absent."""
from dataclasses import dataclass

@dataclass
class HERPV3Config:
    future_horizon: int = 32
    boundary_percentile: float = .90
    boundary_lambda_policy: float = 1.
    boundary_lambda_state: float = 0.
    boundary_min_chain_len: int = 2
    boundary_score_buffer: int = 4096
    chain_radius: float = 2.  # matches scripts/train_v3.py CLI default
    max_regions: int = 256
    centroid_tau: float = .05
    action_feature_weight: float = 1.
    min_common_steps: int = 8
    sigma_floor: float = 1e-3
    sigma_ema_tau: float = .9
    max_sigma_fragments_per_region: int = 32
    max_sigma_policy_lag: int = 2
    sigma_predictor_kappa: float = 8.
    predictor_enabled: bool = True
    predictor_ridge: float = 1e-3
    predictor_min_labels: int = 16
    predictor_refit_every: int = 1
    relevance_ema_tau: float = .9
    relevance_floor: float = 1e-3
    relevance_alpha: float = 1.
    relevance_mode: str = 'cosine'
    max_snapshots_per_region: int = 16
    min_non_root_regions: int = 8
    log_root_total_variance: bool = True
    # p and sigma live on incompatible scales (cosine in [0,1+eps], sigma in
    # sqrt(feature variance)). Rank-normalize each to [0,1] before multiplying
    # so neither factor swamps the other. THEORY §22 preserves ordering.
    score_normalize: str = 'rank'  # {'none','rank','zscore'}
    score_temperature: float = 1.
    # EXPERIMENTS §26 root-handling ablations. Both default to 0 so V3 ("root as
    # ordinary candidate", THEORY §13) is the main-method behavior. Set nonzero
    # to test V1 (hard root lower bound n_0 >= root_floor) or V2 (mix priority
    # with uniform to prevent starvation of cold regions AND root).
    root_floor: float = 0.15  # min fraction of allocation reserved for region 0
    uniform_mix: float = 0.0  # convex mix weight with uniform over all regions
