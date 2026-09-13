# HERP v3 — IMPLEMENTATION.md

> **Purpose:** coding specification for refactoring the current HERP repository to the HERP v3 theory in `THEORY.md`.  
> **Repository:** `Thanh124pav/herp`  
> **Target:** ICRA 2027 experiments.  
> **Priority:** mechanism correctness before scale.

---

# 0. What changes in v3

The repository already contains PPO infrastructure, restorable snapshots, a region archive, gradient signatures, relevance scoring, sigma probing, and a budget allocator.

HERP v3 changes the conceptual core:

1. replace state-radius regions with **trajectory-chain regions**;
2. replace discounted compressed future features with a **fixed-\(M\) direct future-dispersion estimator**;
3. add a simple **linear variance predictor** \(f\);
4. reserve `region_id=0` for the **root / initial-state distribution**;
5. remove hard-coded `uniform_mix`, `staleness_mix`, and permanent `n_min` from the main method;
6. keep gradient cosine alignment as the default \(p_v\);
7. allocate with exactly

   \[
   n_v\propto p_v\sigma_v.
   \]

HERP remains an acquisition layer over PPO. Do not rewrite the PPO optimizer.

---

# 1. Target module layout

Use this layout.

```text
src/herp/
    archive.py
    allocator.py
    chain_features.py        # NEW
    chain_partition.py       # NEW
    region_graph.py          # NEW
    sigma.py                 # REWRITE main path
    sigma_predictor.py       # NEW
    acquisition.py           # NEW
    relevance.py
    gradient_signature.py
    rollout_buffer.py
    logging.py
    envs/
        base.py
        maniskill.py
        metaworld.py
        fetch.py

analysis/
    mechanisms.py
    partition_diagnostics.py # NEW
    sigma_diagnostics.py     # NEW

tests/
    test_chain_partition.py  # NEW
    test_sigma_v3.py         # NEW
    test_sigma_predictor.py  # NEW
    test_root_region.py      # NEW
    test_allocator_v3.py     # NEW
```

Keep the current state-radius regionizer as a legacy ablation, but the main `herp` method must no longer call it.

---

# 2. Global configuration

Create one v3 config dataclass.

```python
@dataclass
class HERPV3Config:
    # fixed future context
    future_horizon: int = 32

    # temporal segmentation
    boundary_percentile: float = 0.90
    boundary_lambda_policy: float = 1.0
    boundary_lambda_state: float = 0.0
    boundary_min_chain_len: int = 2
    boundary_score_buffer: int = 4096

    # cross-trajectory clustering
    chain_radius: float = 0.75
    max_regions: int = 256
    centroid_tau: float = 0.05

    # trajectory metric
    action_feature_weight: float = 1.0
    min_common_steps: int = 8

    # sigma
    sigma_floor: float = 1e-3
    sigma_ema_tau: float = 0.9
    max_sigma_fragments_per_region: int = 32
    max_sigma_policy_lag: int = 2
    sigma_predictor_kappa: float = 8.0

    # linear predictor
    predictor_enabled: bool = True
    predictor_ridge: float = 1e-3
    predictor_min_labels: int = 16
    predictor_refit_every: int = 1

    # relevance
    relevance_ema_tau: float = 0.9
    relevance_floor: float = 1e-3
    relevance_alpha: float = 1.0
    relevance_mode: str = "cosine"

    # archive
    max_snapshots_per_region: int = 16

    # warm-up
    min_non_root_regions: int = 8

    # diagnostics
    log_root_total_variance: bool = True
```

Do not put the following in the main v3 path:

```text
uniform_mix
staleness_mix
permanent n_min
rank-normalized sigma
```

They can remain available only for legacy ablations.

---

# 3. Core data structures

## 3.1 Chain

```python
@dataclass
class Chain:
    episode_id: int
    start_t: int
    end_t: int

    state_features: torch.Tensor
    actions: torch.Tensor

    action_mean: torch.Tensor
    action_logstd: torch.Tensor

    buffer_indices: torch.Tensor

    entry_snapshot: object | None
    entry_feature: torch.Tensor | None = None
```

A chain begins at a detected behavioral boundary and ends immediately before the next boundary.

---

## 3.2 Region

Extend the current `Region` dataclass.

```python
@dataclass
class Region:
    region_id: int
    centroid: torch.Tensor
    is_root: bool = False

    snapshots: list[Snapshot] = field(default_factory=list)
    snapshot_count: int = 0

    num_chains: int = 0
    num_entries: int = 0
    mean_chain_len: float = 0.0

    mean_policy_change: float = 0.0
    mean_action_entropy: float = 0.0
    mean_state_change: float = 0.0

    q_direct: float = float("nan")
    q_pred: float = 0.0
    q_combined: float = 0.0
    sigma_raw: float = 0.0
    sigma_ema: float = 0.0
    sigma_sample_count: int = 0

    p_raw: float = 0.0
    p_ema: float = 0.0

    priority: float = 0.0
    allocated_fragments: int = 0

    last_seen_step: int = 0
    last_scored_step: int = 0
```

Reserve `region_id=0` for root:

```python
root = Region(
    region_id=0,
    centroid=torch.empty(0),
    is_root=True,
)
```

All learned chain regions start at ID `1`.

---

# 4. State and action normalization

Move the reusable running normalization logic into `chain_features.py`.

```python
class RunningFeatureNormalizer:
    def update(self, x: torch.Tensor) -> None:
        ...

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Pipeline:

```python
z_raw = adapter.region_features(obs)
state_normalizer.update(z_raw)
z = state_normalizer.normalize(z_raw)
```

Maintain a separate action normalizer if action dimensions have different scales.

Never use raw observation scale inside the chain Euclidean metric.

---

# 5. Policy-distribution divergence

For diagonal Gaussian PPO, implement exact KL.

```python
def diagonal_gaussian_kl(
    mu_p: torch.Tensor,
    logstd_p: torch.Tensor,
    mu_q: torch.Tensor,
    logstd_q: torch.Tensor,
) -> torch.Tensor:
    # return one scalar per leading batch element
    ...
```

Then:

```python
def symmetric_gaussian_kl(...):
    return 0.5 * (
        diagonal_gaussian_kl(p, q)
        + diagonal_gaussian_kl(q, p)
    )
```

Use actor distribution parameters directly. Do not estimate distribution change from sampled actions when mean and log-standard-deviation are available.

---

# 6. Boundary detector

Create `chain_partition.py`.

At timestep \(t\):

```python
policy_change = symmetric_gaussian_kl(
    mu_prev, logstd_prev,
    mu_curr, logstd_curr,
)

state_change = torch.linalg.vector_norm(
    z_curr - z_prev,
    dim=-1,
)
```

Normalize each scalar channel with running statistics.

```python
score = (
    cfg.boundary_lambda_policy * policy_change_norm
    + cfg.boundary_lambda_state * state_change_norm
)
```

Maintain a bounded deque of recent scores and compute

```python
threshold = torch.quantile(
    torch.as_tensor(score_buffer),
    cfg.boundary_percentile,
)
```

Boundary condition:

```python
is_boundary = (
    score > threshold
    and current_chain_len >= cfg.boundary_min_chain_len
)
```

Episode termination always closes a chain.

If the score buffer is too small to estimate a percentile, do not create an artificial boundary; use episode end only during the short bootstrap phase.

---

# 7. Online chain builder

Maintain one `ChainBuilder` per vectorized env slot.

Pseudo-code:

```python
for env_id in range(num_envs):
    builder = builders[env_id]

    if builder.empty:
        builder.start(current_transition)

    boundary = detector(...)

    if boundary and builder.length >= min_chain_len:
        finished = builder.close_before_current()
        process_chain(finished)
        builder.start_from_current()

    builder.append(current_transition)

    if terminated or truncated:
        finished = builder.close()
        process_chain(finished)
        builder.reset()
```

The chain builder must retain the restorable snapshot at chain entry.

Do not archive random interior snapshots for the main method.

---

# 8. Chain-entry feature and clustering

For a chain beginning at timestep `i`, define

```python
entry_feature = torch.cat([
    z_prev,
    z_start,
], dim=-1)
```

Distance:

```python
def chain_entry_distance(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    d = h1.numel() // 2
    return 0.5 * (
        torch.linalg.vector_norm(h1[:d] - h2[:d])
        + torch.linalg.vector_norm(h1[d:] - h2[d:])
    )
```

Online assignment:

```python
if no_non_root_regions:
    create_region()
else:
    nearest = argmin_distance(entry_feature, region_centroids)

    if min_dist > cfg.chain_radius and num_regions < cfg.max_regions:
        create_region()
    else:
        assign(nearest)
```

Centroid update:

```python
c = (1 - tau) * c + tau * entry_feature
```

Important: the centroid now lives in **chain-entry context space**, not state space.

---

# 9. Snapshot semantics

When a chain is assigned to region `v`, store its entry snapshot:

```python
archive.add_snapshot(
    region_id=v,
    snapshot=chain.entry_snapshot,
)
```

Preserve the existing bounded reservoir sampling behavior.

Main method start position:

```text
chain entry
```

Ablations may use:

```text
random inside chain
chain end
```

but those must not be mixed into the main archive.

---

# 10. Region graph

Create `region_graph.py`.

```python
class RegionGraph:
    def __init__(self):
        self.edge_counts: dict[tuple[int, int], int] = {}

    def observe_path(self, region_ids: list[int]) -> None:
        ...

    def children(self, region_id: int) -> list[int]:
        ...

    def child_probabilities(self, region_id: int) -> dict[int, float]:
        ...
```

For an episode region path:

```text
0 -> 5 -> 5 -> 12 -> 8
```

collapse consecutive duplicates to

```text
0 -> 5 -> 12 -> 8
```

before updating counts.

---

# 11. Fixed-\(M\) rollout fragment

Create a single acquisition fragment structure.

```python
@dataclass
class RolloutFragment:
    source_region_id: int
    policy_version: int
    fragment_id: int

    states: torch.Tensor
    state_features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor

    terminated: torch.Tensor
    truncated: torch.Tensor
    valid_mask: torch.Tensor

    bootstrap_value: torch.Tensor
    entry_snapshot_id: int | None
```

Every scheduled fragment aims for exactly

```python
M = cfg.future_horizon
```

transitions.

If true environment termination occurs early:

- stop stepping that job;
- pad only for tensor storage;
- set `valid_mask=False` for padding;
- all metrics and PPO loss ignore padded entries.

Artificial HERP horizon truncation at step \(M\) uses critic bootstrap.

---

# 12. Root acquisition

When allocator chooses region `0`:

```python
obs, info = env.reset(...)
```

then rollout current policy for up to \(M\) transitions.

Root fragments:

- use `source_region_id=0`;
- are valid PPO data;
- update chain partition and region graph;
- provide direct root-sigma samples;
- can discover new chain regions.

There is no permanent `n0 >= k` rule.

---

# 13. Non-root acquisition

When allocator chooses region `v > 0`:

1. sample one archived chain-entry snapshot;
2. restore a vectorized environment slot;
3. verify restored observation;
4. follow the current stochastic PPO policy for \(M\) transitions;
5. tag all transitions with `source_region_id=v`.

Do not force a first-action perturbation in the main method. The current stochastic policy defines the continuation distribution.

Keep explicit first-action perturbation only as a sigma ablation.

---

# 14. Trajectory feature for sigma

Rewrite the main path in `sigma.py`.

For valid step `m`:

```python
y_m = torch.cat([
    normalized_state_feature_m,
    cfg.action_feature_weight * normalized_action_m,
], dim=-1)
```

State-only ablation:

```python
cfg.action_feature_weight = 0.0
```

---

# 15. Fixed-window pair distance

Implement:

```python
def fixed_window_distance(
    x: RolloutFragment,
    y: RolloutFragment,
    min_common_steps: int,
) -> torch.Tensor | None:
    common = x.valid_mask & y.valid_mask

    if int(common.sum()) < min_common_steps:
        return None

    dx = x.traj_features[common] - y.traj_features[common]
    return dx.square().sum(dim=-1).mean()
```

Default:

```python
min_common_steps = max(2, cfg.future_horizon // 4)
```

Log the fraction of excluded fragment pairs.

For the cleanest mechanism experiments, prefer environments/checkpoints where most fragments reach the full \(M\) steps.

---

# 16. Direct \(q_v\) estimator

Implement:

```python
def direct_q_estimate(
    fragments: list[RolloutFragment],
    min_common_steps: int,
) -> tuple[float, int]:
    """Estimate q = 0.5 E[d_M(X, X')]."""
```

If fewer than two usable fragments exist:

```python
return float("nan"), 0
```

Do not return zero. Zero means observed deterministic continuation, whereas NaN means insufficient evidence.

For usable unordered pairs:

```python
pair_dists = []

for i in range(K):
    for j in range(i + 1, K):
        d = fixed_window_distance(...)
        if d is not None:
            pair_dists.append(d)

if not pair_dists:
    return float("nan"), 0

q_hat = 0.5 * torch.stack(pair_dists).mean()
return float(q_hat), len(pair_dists)
```

For full equal-length fragments, this is the unbiased U-statistic from `THEORY.md`.

---

# 17. Vectorized pairwise sigma

Correctness first. A Python pair loop is acceptable for unit tests and small `K`.

Then optionally vectorize:

```python
# [K, M, D]
features = ...

# [K, K, M, D]
diff = features[:, None, :, :] - features[None, :, :, :]

# [K, K, M]
d2 = diff.square().sum(dim=-1)
```

Apply pair-valid masks and upper-triangular selection.

Do not optimize until profiling shows this code matters.

---

# 18. Recent sigma cache

Future dispersion is policy-dependent. Do not accumulate fragments forever.

Maintain:

```python
region_sigma_fragments: dict[int, deque[RolloutFragment]]
```

with maximum size:

```python
cfg.max_sigma_fragments_per_region
```

Every fragment stores `policy_version`.

Only use fragments satisfying

```python
current_policy_version - fragment.policy_version <= cfg.max_sigma_policy_lag
```

for the direct estimator.

Default lag:

```text
2 PPO updates
```

---

# 19. Linear variance predictor

Create `sigma_predictor.py`.

No neural network in the main v3 method.

```python
class LinearVariancePredictor:
    def __init__(self, feature_dim: int, ridge: float):
        self.feature_dim = feature_dim
        self.ridge = ridge
        self.X = []
        self.y = []
        self.weight = []
        self.coef = None

    def add_label(self, x, q_hat, weight):
        ...

    def fit(self):
        ...

    def predict(self, x):
        ...
```

Use weighted ridge closed form:

\[
\hat\omega=(X^\top WX+\lambda I)^{-1}X^\top Wy.
\]

Implementation:

```python
A = X.T @ W @ X + ridge * I
b = X.T @ W @ y
coef = torch.linalg.solve(A, b)
```

Use `float64` for the solve.

Do not regularize the intercept:

```python
I[0, 0] = 0.0
```

---

# 20. Predictor feature vector

Implement:

```python
def region_predictor_features(region: Region) -> torch.Tensor:
    return torch.tensor([
        1.0,
        region.mean_policy_change,
        region.mean_action_entropy,
        region.mean_state_change,
        math.log1p(region.num_entries),
    ], dtype=torch.float64)
```

Normalize non-intercept features across the predictor dataset.

Do not include:

- chain length;
- current `q_direct`;
- current allocator priority;
- future outcome values not available before acquisition.

---

# 21. Predictor labels

After a region receives fresh allocation, compute a label only from the **fresh fragments generated in that round**.

```python
fresh = fresh_fragments_by_source[region_id]

if len(fresh) >= 2:
    q_hat, pair_count = direct_q_estimate(...)

    if math.isfinite(q_hat):
        predictor.add_label(
            x=region_predictor_features(region),
            q_hat=q_hat,
            weight=float(len(fresh)),
        )
```

Do not repeatedly relabel old cached fragments every iteration.

One allocation round produces at most one new predictor label per region.

---

# 22. Predictor initialization

Before `predictor_min_labels` valid labels exist:

```python
if any_direct_q_exists:
    q_pred = median_recent_direct_q
else:
    q_pred = 0.0
```

This gives a common prior before the linear model is estimable.

It is not a special allocation rule.

---

# 23. Combine direct and predicted variance

Implement:

```python
def combined_q(
    q_direct: float,
    q_pred: float,
    direct_sample_count: int,
    kappa: float,
) -> float:
    q_pred = max(0.0, q_pred)

    if not math.isfinite(q_direct):
        return q_pred

    q_direct = max(0.0, q_direct)
    lam = direct_sample_count / (direct_sample_count + kappa)

    return lam * q_direct + (1.0 - lam) * q_pred
```

Then:

```python
q = combined_q(...)
sigma_alloc = math.sqrt(
    max(0.0, q) + cfg.sigma_floor ** 2
)
```

Update region fields:

```python
region.q_direct = q_direct
region.q_pred = q_pred
region.q_combined = q
region.sigma_raw = sigma_alloc
region.sigma_ema = (
    tau * region.sigma_ema
    + (1 - tau) * sigma_alloc
)
```

Initialize the first EMA observation directly rather than shrinking from zero.

---

# 24. Root sigma

Root fragments are still stored so that direct root dispersion can be measured, but the **main root sigma used by the allocator is the total-variance aggregation over root child regions**. Direct root sigma is a diagnostic/oracle-like validation signal.

Required per child:

```text
child probability w_c
child mean future representation mu_c
child q_c
```

Then:

```python
mu0 = sum(w[c] * mu[c] for c in children)

q0_total = sum(
    w[c] * (
        q[c]
        + torch.sum((mu[c] - mu0) ** 2).item()
    )
    for c in children
)
```

For each child, use its current combined within-region estimate `q_combined` and a recent empirical mean future representation `mu_c`. If there are no observed children, use empirical root variance `0.0`; the common sigma floor then keeps root alive.

Set:

```python
root.q_combined = q0_total
root.sigma_raw = math.sqrt(q0_total + cfg.sigma_floor**2)
```

Log in parallel:

```text
sigma/root_direct          # validation only
sigma/root_total_variance  # main root score
sigma/root_abs_gap
```

---

# 25. Region statistics for predictor

When assigning a new chain to region `v`, update stable online means for:

```python
region.mean_policy_change
region.mean_action_entropy
region.mean_state_change
region.mean_chain_len
region.num_entries
region.num_chains
```

For a diagonal Gaussian actor, entropy can be computed analytically.

If PPO uses a global state-independent `logstd`, entropy may contain little information. That is acceptable; the linear coefficient should reveal this.

---

# 26. Relevance \(p_v\)

Keep `relevance.py` but simplify the main path to cosine gradient alignment.

Current utility:

```python
def cosine_relevance(
    region_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
) -> float:
    ...
```

Main tracker update:

```python
raw = cosine_relevance(g_v, g_ref)
positive = max(0.0, raw)

region.p_raw = raw
region.p_ema = (
    cfg.relevance_ema_tau * region.p_ema
    + (1 - cfg.relevance_ema_tau) * positive
)

p_alloc = (
    region.p_ema + cfg.relevance_floor
) ** cfg.relevance_alpha
```

Do not separately normalize `p_alloc` across regions.

The final allocator normalizes `p*sigma`.

---

# 27. Reference batch

Reference data must come only from ordinary environment resets.

At relevance refresh:

1. reset reference envs normally;
2. roll out the current policy;
3. compute GAE using the same conventions as main PPO;
4. compute actor gradient signature `g_ref`.

Reference interactions count toward the training interaction budget unless the paper explicitly reports them separately.

Do not build `g_ref` from restored HERP states.

---

# 28. Regional gradient signatures

For source region `v`, use transitions whose

```python
source_region_id == v
```

and compute

```python
g_v = gradient_signature(region_batch)
```

Use the full actor parameter set.

Advantage normalization must be consistent between `g_v` and `g_ref`.

Do not independently zero-center tiny regional batches if that rotates the gradient relative to the PPO update actually performed.

Prefer one shared scoring-round advantage scale.

---

# 29. Root relevance

Root relevance is measured, not hard-coded.

Root transitions have:

```python
source_region_id = 0
```

Compute:

```python
g0 = gradient_signature(root_batch)
p0_raw = cosine_relevance(g0, g_ref)
```

Then apply the same EMA and floor as all other regions.

---

# 30. Allocator rewrite

Rewrite the main `allocator.py` path.

Remove from the main score:

```text
robust rank transform of sigma
uniform_mix
staleness_mix
n_min
```

Compute:

```python
p = (
    max(0.0, region.p_ema)
    + cfg.relevance_floor
) ** cfg.relevance_alpha

sigma = max(
    cfg.sigma_floor,
    region.sigma_ema,
)

score = p * sigma
```

Then:

```python
scores = torch.tensor([...], dtype=torch.float64)

if not torch.isfinite(scores).all():
    raise FloatingPointError("non-finite HERP v3 priority")

assert torch.all(scores > 0)
priority = scores / scores.sum()
```

Do not code a normal-case fallback:

```python
if scores.sum() == 0:
    uniform()
```

Positive floors make that unnecessary.

---

# 31. Integer allocation

For a round budget `B_round`:

```python
num_fragments = B_round // cfg.future_horizon
```

Sample:

```python
draws = torch.multinomial(
    priority,
    num_samples=num_fragments,
    replacement=True,
    generator=generator,
)

counts = torch.bincount(
    draws,
    minlength=len(active_regions),
)
```

Return:

```python
{
    region.region_id: int(counts[i])
    for i, region in enumerate(active_regions)
}
```

A largest-remainder deterministic allocator may be used for mechanism tests.

---

# 32. Acquisition scheduler

Create `acquisition.py`.

```python
@dataclass
class AcquisitionJob:
    region_id: int
    snapshot: Snapshot | None
    max_steps: int
```

```python
class AcquisitionScheduler:
    def schedule(
        self,
        regions: list[Region],
        allocation: dict[int, int],
        num_envs: int,
    ) -> list[AcquisitionJob]:
        ...
```

Root job:

```python
AcquisitionJob(
    region_id=0,
    snapshot=None,
    max_steps=M,
)
```

Non-root job samples one region snapshot.

Fill vectorized slots, execute, then continue scheduling until all fragment counts are exhausted.

---

# 33. Rollout buffer metadata

Add:

```python
source_region_id: torch.Tensor
chain_region_id: torch.Tensor
acquisition_round: torch.Tensor
fragment_id: torch.Tensor
fragment_step: torch.Tensor
is_restored_start: torch.Tensor
```

Definitions:

- `source_region_id`: region where this \(M\)-step fragment started;
- `chain_region_id`: behavior-chain region currently assigned to this transition.

Do not conflate them.

`p_v` groups by `source_region_id`.

Partition and region graph use `chain_region_id` / chain sequence.

---

# 34. PPO GAE across fragment boundaries

GAE must not leak across independent restored fragments.

For each fragment:

```python
if true_termination:
    next_nonterminal = 0.0
    next_value = 0.0
else:
    next_nonterminal = 1.0
    next_value = critic(last_obs)
```

Run reverse-time GAE only inside the fragment.

Afterwards concatenate all valid fragment transitions into ordinary PPO minibatches.

Artificial truncation at HERP step \(M\) is bootstrapped.

---

# 35. Policy versioning

Increment:

```python
policy_version += 1
```

after each PPO update.

Each rollout fragment stores the version that generated it.

Direct sigma uses only recent policy versions.

The predictor may retain older labels, but log label age. A later ablation can apply exponential age weighting.

---

# 36. Warm-up

HERP cannot score regions before any regions exist.

Use a data-availability warm-up:

```python
while (
    num_non_root_regions < cfg.min_non_root_regions
    or num_sigma_labels < cfg.predictor_min_labels
):
    allocate_root_only()
```

This is not a permanent root lower bound. It only instantiates the objects required by the method.

Log:

```text
herp_activation_step
```

---

# 37. Main training-round ordering

Use this ordering to avoid label/policy mismatch.

```text
policy version k
    ↓
score current regions using version-k evidence
    ↓
allocate
    ↓
collect version-k fresh fragments
    ↓
partition + archive + build fresh sigma labels
    ↓
PPO update
    ↓
policy version k+1
```

The fresh fragments generated by version \(k\) become labels for future scoring.

Do not recompute a label from trajectories that were not generated by the policy version being analyzed.

---

# 38. End-to-end pseudo-code

```python
def train_herp_v3(...):
    init_ppo()
    init_root_region()
    init_chain_partition()
    init_region_graph()
    init_sigma_predictor()

    # ---- warm-up ----
    while not data_ready():
        fragments = acquire_root_fragments(...)
        process_partition(fragments)
        add_fresh_sigma_labels(fragments)
        ppo_update(fragments)
        policy_version += 1

    # ---- HERP rounds ----
    while global_train_steps < total_timesteps:

        # A. reference objective
        reference = collect_reference_batch()
        g_ref = gradient_signature(reference)

        # B. score active regions
        for region in active_regions:
            region.p_raw = score_relevance(region, g_ref)
            update_relevance_ema(region)

            if not region.is_root:
                q_direct, n_direct = estimate_recent_q(region)

                x = region_predictor_features(region)
                q_pred = predictor.predict_or_prior(x)

                q = combined_q(
                    q_direct=q_direct,
                    q_pred=q_pred,
                    direct_sample_count=n_direct,
                    kappa=cfg.sigma_predictor_kappa,
                )

                sigma = math.sqrt(
                    max(0.0, q) + cfg.sigma_floor**2
                )

                update_region_sigma_fields(region, q_direct, q_pred, q, sigma)

        # root sigma is recursively induced by child regions
        update_root_total_variance(root, region_graph, active_regions)

        # C. allocate
        probs = priority_distribution(active_regions, cfg)
        allocation = allocate_fragments(probs, round_budget)

        # D. acquire with current policy
        fresh_fragments = scheduler.execute(allocation)

        # E. partition all newly observed experience
        chains = chain_partitioner.process(fresh_fragments)
        assigned = chain_regionizer.assign(chains)
        update_archive(assigned)
        region_graph.observe(...)

        # F. create fresh direct q labels before discarding grouping
        add_fresh_sigma_labels(fresh_fragments)

        # G. PPO
        ppo_update(fresh_fragments)
        policy_version += 1

        # H. refit simple predictor
        if predictor.ready() and should_refit():
            predictor.fit()

        # I. log/checkpoint
        write_logs()
        maybe_checkpoint()
```

---

# 39. Partition mechanism test

Create `analysis/partition_diagnostics.py`.

For validation trajectories, log and plot:

- policy-distribution change score;
- state-change score;
- percentile threshold;
- detected boundary locations;
- chain IDs;
- task events if available.

Required summary metrics:

```text
chains per episode
chain length distribution
regions per 100k env steps
entry count per region
```

Add a synthetic invariance test for the motivating example:

- generate or transform a trajectory with identical behavioral mode;
- multiply one monotonically changing state coordinate by `1`, `10`, `100`, `1000`;
- after normalization, the number of behavioral chains should not grow proportionally with scale/path length.

---

# 40. Sigma mechanism test

At frozen policy checkpoints:

1. choose at least 30 regions;
2. acquire low-budget continuations;
3. acquire high-budget oracle-like continuations;
4. compare estimators.

Defaults:

```text
K_low = 4
K_high = 64
```

Compute:

```text
q_direct_low
q_direct_high
q_pred
q_combined
```

Report:

- Pearson correlation;
- Spearman correlation;
- bootstrap 95% CI;
- normalized MAE;
- top-k overlap.

The prior pilot correlation around `0.63` is useful historical context, not a hard pass criterion.

---

# 41. Predictor mechanism test

Use temporal or region holdout.

Train the linear predictor on one subset of labels and evaluate against high-sample \(q_v\) on held-out regions.

Report:

```text
R^2
Spearman rho
MAE
calibration plot
```

Also report coefficients:

```text
intercept
policy_change
action_entropy
state_change
log_visit_count
```

If the linear model does not improve finite-sample ranking over direct estimation, keep it as an ablation instead of forcing it into the main method.

---

# 42. Relevance mechanism test

Retain the controlled PPO-delta protocol.

For region \(v\):

1. clone current PPO policy into models A and B;
2. give both the same matched base PPO update;
3. give B additional matched region-\(v\) data;
4. evaluate both on ordinary-reset reference episodes;
5. compute

   \[
   \Delta_v=J_{\rm ref}(B)-J_{\rm ref}(A).
   \]

Measure correlation between cosine \(p_v\) and \(\Delta_v\).

Do not use the old one-step SGD proxy as the mechanism oracle.

---

# 43. Partition unit tests

Create `tests/test_chain_partition.py`.

### Constant policy regime

Large state drift, unchanged policy distribution.

Expected:

```text
one chain until episode end
```

### Policy switch

State evolves smoothly but policy mean changes sharply.

Expected:

```text
boundary near switch
```

### Scale normalization

Multiply one raw state coordinate by `1000`.

Expected:

```text
partition approximately unchanged after normalization
```

### Entry-context clustering

Two long chains have similar `(pre_state, start_state)` but very different endpoints.

Expected:

```text
same region if entry distance < chain_radius
```

This protects the exact intended 1 m / 1000 m behavior.

---

# 44. Sigma unit tests

Create `tests/test_sigma_v3.py`.

### Identical futures

```python
q_hat == 0.0
sigma_alloc == sigma_floor
```

### Insufficient data

One fragment:

```python
math.isnan(q_hat)
```

not zero.

### Permutation invariance

Reordering continuation fragments must not change `q_hat`.

### Horizon fairness

Two regions with identical first \(M\) future behavior but different total chain lengths must have equal `q_hat`.

### Unbiased Monte Carlo

Construct a synthetic distribution with known covariance trace and verify across repeated trials:

```text
mean(q_hat) ≈ q_oracle
```

within statistical tolerance.

---

# 45. Predictor unit tests

Create `tests/test_sigma_predictor.py`.

Synthetic realizable target:

```python
q = 2.0 + 0.5 * x1 - 0.25 * x2 + noise
```

with zero-mean noise.

Verify:

- estimated coefficients approach the true values as sample count grows;
- ridge solve remains finite;
- intercept is not regularized;
- non-negative inference clipping works;
- confidence weights change the weighted fit as expected.

Also include a non-realizable target and ensure tests only require convergence to the best linear fit.

---

# 46. Root unit tests

Create `tests/test_root_region.py`.

### Root exists

```python
assert regions[0].region_id == 0
assert regions[0].is_root
```

### No permanent root lower bound

Synthetic priorities may legitimately yield:

```text
n0 = 0
```

for one finite allocation round.

### Degenerate limit

Set all raw `p` to zero and all sigma to the common floor.

Expected:

```text
priority == uniform
```

within tolerance.

### Root acquisition

Root job must call environment reset and must not call snapshot restore.

---

# 47. Allocator unit tests

Create `tests/test_allocator_v3.py`.

Analytic priority cases:

```text
p=[1,1], sigma=[1,1] -> [0.5, 0.5]
p=[2,1], sigma=[1,1] -> [2/3, 1/3]
p=[1,1], sigma=[3,1] -> [3/4, 1/4]
p=[2,1], sigma=[3,1] -> [6/7, 1/7]
```

No sigma rank transform is allowed in the main path.

---

# 48. Budget accounting

Every training-time environment transition belongs to exactly one category:

```text
ROOT_ACQUISITION
REGION_ACQUISITION
REFERENCE
```

Track:

```python
global_train_steps = (
    root_acquisition_steps
    + region_acquisition_steps
    + reference_steps
)
```

Assertions every round:

```python
assert (
    root_steps
    + region_steps
    + reference_steps
    == global_train_steps
)

assert global_train_steps <= total_timesteps
```

No sigma or relevance rollout is free.

Evaluation-only episodes may be tracked separately if the paper explicitly excludes evaluation interaction from training budget.

---

# 49. Logging schema

Global metrics:

```text
herp/num_regions
herp/num_chains
herp/mean_chain_len
herp/boundary_threshold

allocation/root_fraction
allocation/entropy
allocation/max_region_fraction

sigma/direct_mean
sigma/pred_mean
sigma/combined_mean
sigma/root_direct
sigma/root_total_variance
sigma/floor_fraction

predictor/num_labels
predictor/r2_train
predictor/coef_policy_change
predictor/coef_action_entropy
predictor/coef_state_change
predictor/coef_log_visit_count

relevance/mean
relevance/root
relevance/positive_fraction

budget/root_steps
budget/region_steps
budget/reference_steps
budget/global_steps
```

Per-region table:

```text
step
policy_version
region_id
is_root
num_entries
mean_chain_len
p_raw
p_ema
q_direct
q_pred
q_combined
sigma
priority
allocated_fragments
```

---

# 50. Checkpoint state

Checkpoint must preserve:

```python
{
    "policy": ...,
    "critic": ...,
    "optimizer": ...,

    "state_normalizer": ...,
    "action_normalizer": ...,
    "boundary_normalizer": ...,
    "boundary_score_buffer": ...,

    "archive": ...,
    "region_graph": ...,

    "sigma_fragment_metadata": ...,
    "sigma_predictor_X": ...,
    "sigma_predictor_y": ...,
    "sigma_predictor_weight": ...,
    "sigma_predictor_coef": ...,

    "relevance_tracker": ...,

    "global_train_steps": ...,
    "policy_version": ...,
    "allocation_round": ...,
}
```

Resume must preserve region IDs, especially root ID `0`.

---

# 51. Migration map from current repository

## `src/herp/regions.py`

Current concept:

```text
radius clustering over normalized individual states
```

v3:

```text
keep as legacy baseline
move normalizer utility out
main method uses chain segmentation + chain-entry regionizer
```

## `src/herp/sigma.py`

Current concept:

```text
discounted future feature
pairwise sigma
branch/dynamics decomposition
```

v3:

```text
fixed-M state-action trajectory metric
direct q U-statistic
root total-variance diagnostic
```

Keep old functions under clearly marked legacy/ablation names if needed.

## `src/herp/relevance.py`

Current concept:

```text
cosine + EMA + separate p normalization
```

v3:

```text
keep cosine + EMA
apply positive floor
no separate normalization in main path
```

## `src/herp/allocator.py`

Current concept includes:

```text
sigma rank transform
uniform mix
staleness mix
n_min
```

v3 main path:

```text
score = p_alloc * sigma_alloc
normalize once
sample rollout-unit counts
```

## `src/herp/archive.py`

v3:

```text
reserve region_id=0 for root
store chain-entry snapshots
extend Region statistics
preserve bounded reservoir sampling
```

---

# 52. Experimental coding stages

Do not jump directly to full HERP training.

## Stage A — partition only

Run ordinary PPO and only observe chains/regions.

Deliver:

- boundary plots;
- chain-length distribution;
- region-count curves;
- synthetic 1 m / 1000 m invariance test.

## Stage B — sigma only

Freeze policy checkpoints.

Deliver:

- low-budget versus high-budget fixed-\(M\) correlation;
- variable-length versus fixed-\(M\) comparison;
- predictor ablation.

## Stage C — relevance only

Run controlled PPO-delta test.

Deliver positive/negative correlation result for cosine relevance.

## Stage D — allocator

Compare:

```text
PPO / root only
Uniform region allocation
Sigma only
p only
HERP p*sigma
```

## Stage E — full suite

Only after A-D pass.

---

# 53. Required algorithm baselines

Minimum end-to-end set:

```text
PPO
RND
Disagreement
Uniform-region allocation
HERP-sigma
HERP-p
HERP-p*sigma
```

All must share the same PPO backbone and total environment-interaction accounting.

---

# 54. Failure modes

## Too many regions

Symptoms:

```text
max_regions reached early
median entries per region near 1
```

Check:

- chain radius too small;
- boundary percentile too low;
- feature normalization broken.

Do not immediately add region deletion.

## No boundaries

Symptoms:

```text
one chain per episode
```

Check:

- policy KL implementation;
- whether actor distribution changes with state;
- percentile threshold;
- score normalization.

## Sigma collapse

Symptoms:

```text
q_direct ≈ 0 for most regions
```

Check:

- deterministic policy sampling accidentally enabled;
- vectorized rollouts share identical RNG state;
- snapshot restore duplicates RNG state;
- action normalization;
- state feature sensitivity.

## Sigma explosion

Check:

- unnormalized state dimensions;
- padding included as real data;
- stale policy fragments;
- one action dimension dominating the trajectory metric.

## Relevance collapse

If all raw cosine values are non-positive, positive floors keep allocation mathematically defined.

Log the event. Do not switch algorithms through an exception branch.

## Root starvation

`n0=0` in one round is valid.

Persistent root starvation should be diagnosed through:

```text
root p
root sigma
root allocation fraction
new-region discovery rate
```

Do not add a permanent root quota without an ablation showing it is necessary.

---

# 55. Reproducibility

Log for every run:

```text
git commit SHA
seed
task
environment version
torch version
CUDA version
simulator version
num_envs
future_horizon M
boundary percentile
chain radius
sigma floor
relevance floor
predictor ridge
predictor coefficients
total environment interactions
```

Seed Python, NumPy, PyTorch, CUDA, and environment slots independently.

Restored parallel rollouts must not accidentally reuse identical environment RNG streams.

---

# 56. Minimum acceptance gates

Do not launch expensive main experiments until all are true.

1. Constant-behavior long-motion test produces approximately one chain.
2. Known action-regime switch produces a boundary.
3. Cross-trajectory clustering is insensitive to interior chain length.
4. Direct \(\hat q\) passes a synthetic unbiasedness test.
5. Fixed-\(M\) low-budget estimate positively correlates with high-budget oracle.
6. Linear predictor does not catastrophically reduce held-out rank correlation.
7. Cosine relevance has positive controlled PPO-delta correlation.
8. All-zero raw signals produce uniform allocation through common floors.
9. Root may receive zero finite-round allocation without crashing.
10. Selecting root actually resets the environment.
11. Every training interaction is counted exactly once.
12. Disabling HERP acquisition reproduces the PPO baseline behavior.

---

# 57. Recommended coding order

Implement in this order:

```text
1. Chain / Region dataclasses
2. Gaussian policy divergence
3. Boundary detector
4. Online ChainBuilder
5. Chain-entry clustering
6. Root region + region graph
7. Fixed-M RolloutFragment
8. Direct q estimator + unit tests
9. Linear variance predictor + unit tests
10. Relevance main-path simplification
11. Allocator rewrite
12. Acquisition scheduler
13. Fragment-local PPO GAE
14. Mechanism diagnostics
15. End-to-end pilot
```

Do not begin from the allocator. The allocator is the simplest component; correctness depends on the partition and sigma objects first.

---

# 58. Debug-output contract

A debug run should expose something structurally like:

```text
step=250000
regions=21
chains=4382

root:
    p=0.71
    q_direct=0.083
    q_pred=0.074
    sigma=0.284
    allocation=7/64

region_4:
    p=0.66
    q_direct=0.211
    q_pred=0.192
    sigma=0.452
    allocation=10/64

region_9:
    p=0.19
    q_direct=0.330
    q_pred=0.284
    sigma=0.557
    allocation=5/64

allocation_entropy=2.41
global_train_steps=250000
budget_check=PASS
```

Exact values are arbitrary. Required semantics are:

- root appears in the same region table;
- there is no hard-coded root quota;
- direct and predicted variance are separately visible;
- allocation follows \(p\sigma\);
- budget accounting passes.

---

# 59. Final implementation principle

Keep the ICRA method readable as

```text
behavioral chain partition
        ↓
fixed-M future dispersion σ
        ↓
gradient alignment p
        ↓
allocation ∝ pσ
        ↓
PPO
```

The linear predictor is deliberately low-capacity. Do not add a learned world model, learned region encoder, Hessian influence estimator, or another exploration bonus to the main HERP v3 path before the minimal method is validated.
