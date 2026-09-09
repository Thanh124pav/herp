# HERP IMPLEMENTATION.md

## 0. Goal

Implement one simple ICRA-ready HERP codebase with:

1. one PPO backbone,
2. one HERP core,
3. three benchmark adapters,
4. six main methods,
5. strict equal-budget accounting,
6. reproducible experiment runners.

Supported benchmark groups:

- ManiSkill
- Meta-World
- Gymnasium Robotics / Fetch

Do not turn HERP into a general robotics framework.

---

# 1. Repository structure

```text
herp/
├── train.py
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── maniskill/
│   ├── metaworld/
│   └── fetch/
├── src/herp/
│   ├── allocator.py
│   ├── archive.py
│   ├── gradient_signature.py
│   ├── probe.py
│   ├── regions.py
│   ├── relevance.py
│   ├── rollout_buffer.py
│   ├── sigma.py
│   ├── eval.py
│   ├── logging.py
│   ├── baselines/
│   │   ├── rnd.py
│   │   └── disagreement.py
│   └── envs/
│       ├── base.py
│       ├── maniskill.py
│       ├── metaworld.py
│       └── fetch.py
├── scripts/
│   ├── run_suite.py
│   ├── run_mechanisms.py
│   ├── run_ablation.py
│   ├── check_restore.py
│   └── smoke.py
└── analysis/
    ├── aggregate.py
    ├── plot_learning.py
    ├── plot_sigma.py
    └── plot_p.py
```

The old `experience_routing/` code may remain temporarily, but new HERP experiments must import only `src/herp`.

---

# 2. Minimal EnvAdapter

Create:

```python
from abc import ABC, abstractmethod

class EnvAdapter(ABC):

    @abstractmethod
    def reset(self, seed=None):
        ...

    @abstractmethod
    def step(self, action):
        ...

    @abstractmethod
    def save_state(self):
        ...

    @abstractmethod
    def restore_state(self, snapshot):
        ...

    @abstractmethod
    def obs_tensor(self, obs):
        ...

    @abstractmethod
    def region_features(self, obs):
        ...

    @abstractmethod
    def action_low(self):
        ...

    @abstractmethod
    def action_high(self):
        ...

    @abstractmethod
    def elapsed_steps(self):
        ...

    @abstractmethod
    def success_from_info(self, info):
        ...
```

HERP core must not directly call ManiSkill- or MuJoCo-specific APIs.

---

# 3. ManiSkill adapter

Wrap the current working ManiSkill implementation.

Snapshot:

```python
env.unwrapped.get_state_dict()
```

Restore using:

```python
env.reset(
    options={
        "reset_to_env_states": {
            "env_states": snapshot.env_state
        }
    }
)
```

Also restore:

- controller state,
- elapsed steps,
- wrapper elapsed-step counters.

Keep the current restoration logic because the pilot already showed near-zero restore error.

Use the current state observation for:

```python
region_features(obs)
```

after normalization.

---

# 4. Meta-World adapter

Support single-task environments.

Representative tasks:

```text
button-press-v3
drawer-open-v3
pick-place-v3
peg-insert-side-v3
```

Use dense reward if available.

Snapshot must contain at least:

```python
@dataclass
class MetaWorldSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    task_state: dict
    elapsed_steps: int
```

`task_state` must preserve task-specific state such as:

- randomized goal,
- object configuration,
- episode-local variables.

Do not assume `qpos/qvel` alone are sufficient.

Restore test:

1. reset,
2. run random actions,
3. save snapshot,
4. apply action `a`,
5. restore,
6. apply same `a`,
7. compare observation, reward, done flags.

Require approximately:

```text
max observation error < 1e-5
reward error < 1e-6
termination/truncation match
```

If exact replay fails, do not use that task in the paper.

---

# 5. Fetch adapter

Support:

```text
FetchPush-v4
FetchPickAndPlace-v4
```

Flatten dict observation as:

```python
obs_tensor = torch.cat([
    observation,
    achieved_goal,
    desired_goal,
])
```

Use the same representation initially for regionization.

Snapshot:

```python
@dataclass
class FetchSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    desired_goal: np.ndarray
    elapsed_steps: int
    extra: dict
```

Restore all simulator and goal-related state.

Run the same one-step replay test as Meta-World.

---

# 6. PPO backbone

Use one PPO implementation everywhere.

Actor:

```text
Linear(obs_dim, 256)
Tanh
Linear(256, 256)
Tanh
Linear(256, action_dim)
```

Critic:

```text
Linear(obs_dim, 256)
Tanh
Linear(256, 256)
Tanh
Linear(256, 1)
```

Default hyperparameters:

```yaml
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_coef: 0.2
update_epochs: 4
num_minibatches: 4
max_grad_norm: 0.5
```

Task-specific horizons and total timesteps may differ.

Do not change the PPO backbone between methods.

---

# 7. Fix gradient signature before long runs

The pilot actor-head-only signature should be replaced.

Use full actor parameters:

```python
def signature_parameters(agent):
    params = []
    params.extend(agent.actor.parameters())
    params.extend(agent.actor_head.parameters())
    params.append(agent.logstd)
    return params
```

Do not include critic parameters.

---

# 8. Fix advantage normalization for \(p_v\)

Do not independently zero-mean and normalize each small region batch.

Recommended:

```python
adv_scale = running_std_of_training_advantages
adv_sig = advantages / (adv_scale + 1e-8)
```

Use the same scale for reference and candidate region signatures in a scoring round.

Do not subtract each region's own advantage mean.

---

# 9. Policy gradient signature

Use

\[
L_{\mathrm{sig}}
=
-
\mathbb E[
\log\pi_\theta(a\mid s)\hat A
].
\]

Implementation:

```python
def policy_gradient_signature(agent, obs, actions, advantages, adv_scale):
    params = signature_parameters(agent)

    dist = agent.get_distribution(obs)
    logprob = dist.log_prob(actions).sum(-1)

    adv = advantages.detach() / (adv_scale + 1e-8)

    loss = -(logprob * adv).mean()

    grads = torch.autograd.grad(
        loss,
        params,
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )

    return flatten(grads)
```

Do not add gradient projection unless memory becomes an actual problem.

---

# 10. Implement \(p_v\) estimators

Support:

```text
cosine
dot
fisher
occupancy
hybrid
```

Cosine:

\[
p_v^{\mathrm{raw}}
=
\frac{
g_{\mathrm{ref}}^\top g_v
}{
\|g_{\mathrm{ref}}\|
\|g_v\|
+\epsilon
}.
\]

Dot:

\[
p_v^{\mathrm{raw}}
=
g_{\mathrm{ref}}^\top g_v.
\]

Diagonal Fisher:

\[
p_v
=
g_{\mathrm{ref}}^\top
\frac{
g_v
}{
F_{\mathrm{diag}}+\lambda
}.
\]

Occupancy:

\[
p_v^{\mathrm{occ}}
=
\frac{
N_v+\epsilon
}{
N_{\mathrm{ref}}+K\epsilon
}.
\]

Hybrid:

```python
score = (
    (p_occ + eps) ** eta
    * (p_grad + eps) ** (1 - eta)
)
```

Use default:

```yaml
eta: 0.5
```

Ablate:

```text
0.0
0.5
1.0
```

---

# 11. Reference batch

Reference rollouts must:

- start from ordinary environment reset,
- use current policy,
- use separate training-time seeds,
- never use final evaluation seeds,
- never be used for PPO optimization.

Recommended:

```yaml
reference_horizon: 128
relevance_interval: 5
```

Reference steps count toward the global interaction budget.

---

# 12. Archive

Keep:

```python
@dataclass
class Region:
    region_id: int
    centroid: Tensor
    count: int
    snapshots: list
    last_seen_step: int
    last_probed_step: int
    sigma_raw: float
    sigma_ema: float
    p_raw: float
    p_ema: float
    priority: float
```

Snapshot:

```python
@dataclass
class Snapshot:
    env_state: Any
    obs: Tensor
    global_step: int
    episode_id: int
    return_so_far: float
    elapsed_steps: int
```

Use reservoir sampling.

Recommended:

```yaml
max_snapshots_per_region: 8
```

---

# 13. Regionizer

Keep radius-based clustering.

Recommended defaults:

```yaml
region_radius: 0.5
max_regions: 128
archive_interval: 8
max_candidates: 4
```

Use:

```python
z = env_adapter.region_features(obs)
```

then HERP's running normalizer.

No learned embedding for ICRA.

---

# 14. \(\sigma_v\) probes

Support:

```text
pairwise
branch
return
```

Default:

```yaml
probe_horizon: 8
num_probes: 4
num_env_repeats: 1
probe_scale: 1.0
gamma_branch: 0.95
lambda_dyn: 0.0
```

Pairwise:

```python
sigma = sqrt(
    0.5 * mean_pairwise_squared_distance(Z)
)
```

For branch decomposition, share continuation noise after the first action.

Probe interactions count toward the training budget.

Simplest ICRA rule:

- probes only score regions,
- allocated rollouts are used for PPO.

Do not silently train PPO on perturbed probes unless their policy likelihood is handled correctly.

---

# 15. Budget accounting

At every round:

```text
normal_steps
+ reference_steps
+ probe_steps
+ allocated_steps
= round_budget
```

Cumulatively:

```text
sum(training interaction categories)
= global_env_steps
<= total_timesteps
```

Evaluation steps use a separate counter.

Keep the current assertions.

---

# 16. Allocator

Use:

```python
score_v = (
    (p_v + eps_p) ** alpha
    * (sigma_v + eps_sigma) ** beta
)
```

Default:

```yaml
alpha: 1.0
beta: 1.0
eps_p: 0.05
eps_sigma: 0.05
uniform_mix: 0.10
staleness_mix: 0.05
```

For HERP-sigma:

```text
p_v = 1
```

For HERP-p:

```text
sigma_v = 1
```

---

# 17. RND baseline

Use same PPO backbone.

Intrinsic reward:

\[
r_t^{\mathrm{total}}
=
r_t^{\mathrm{task}}
+
\lambda_{\mathrm{RND}}
r_t^{\mathrm{RND}}.
\]

Tune only on one development task:

```text
0.001
0.01
0.1
```

Then freeze as much as possible across the suite.

---

# 18. Disagreement baseline

Use a small forward-model ensemble:

\[
f_j(s_t,a_t)
\rightarrow
\hat s_{t+1}.
\]

Intrinsic bonus:

\[
r_t^{\mathrm{dis}}
=
\operatorname{Var}_j[f_j(s_t,a_t)].
\]

Use same PPO backbone and the same coefficient-tuning protocol as RND.

---

# 19. Main methods

Main paper:

```text
ppo
rnd
disagreement
herp_sigma
herp_p
herp
```

Optional:

```text
go_explore
plr
```

Do not block the paper on optional baselines.

---

# 20. Tasks

ManiSkill:

```text
PushCube-v1
PickCube-v1
StackCube-v1
PegInsertionSide-v1
```

Meta-World:

```text
button-press-v3
drawer-open-v3
pick-place-v3
peg-insert-side-v3
```

Fetch:

```text
FetchPush-v4
FetchPickAndPlace-v4
```

Total:

```text
10 tasks
3 benchmark groups
```

---

# 21. Config files

Use:

```text
configs/maniskill/pushcube.yaml
configs/maniskill/pickcube.yaml
configs/maniskill/stackcube.yaml
configs/maniskill/peginsertion.yaml

configs/metaworld/button_press.yaml
configs/metaworld/drawer_open.yaml
configs/metaworld/pick_place.yaml
configs/metaworld/peg_insert_side.yaml

configs/fetch/push.yaml
configs/fetch/pick_place.yaml
```

Task files contain only task-specific settings.

HERP defaults stay in:

```text
configs/base.yaml
```

---

# 22. Main experiment protocol

For every task-method pair:

```text
3 training seeds minimum
5 preferred if compute permits
```

At each evaluation checkpoint:

```text
50 ordinary-reset evaluation episodes
```

Report:

1. success rate,
2. raw or normalized return,
3. success AUC,
4. return AUC,
5. steps-to-threshold where meaningful,
6. wall-clock overhead separately.

---

# 23. The 32k pilot is not a final benchmark

Before performance claims:

1. verify PPO learns each task,
2. choose an adequate training budget,
3. rerun all main methods under equal budget.

If PPO remains at zero success, do not use that budget to rank methods by final success.

---

# 24. \(\sigma_v\) mechanism test

Run on:

```text
ManiSkill PickCube
ManiSkill PegInsertionSide
Meta-World pick-place
```

Procedure:

1. sample 50 archived regions,
2. estimate \(K=4\) sigma,
3. independently estimate \(K=64\) oracle sigma,
4. compute Spearman correlation,
5. save top/bottom branching-state visualizations.

Optional:

\[
\rho(
\sigma_{\mathrm{traj}},
\sigma_{\mathrm{grad}}
).
\]

---

# 25. \(p_v\) mechanism test

Replace the actor-head one-step SGD diagnostic.

For each sampled region:

1. clone checkpoint twice,
2. collect a matched base PPO batch,
3. collect region batch,
4. update copy A on base,
5. update copy B on base + region,
6. evaluate both on identical reference seeds.

Define:

\[
\Delta_v
=
J_{\mathrm{ref}}(\theta_B)
-
J_{\mathrm{ref}}(\theta_A).
\]

Correlate \(\Delta_v\) with:

```text
occupancy
cosine
dot
fisher
hybrid
```

Use at least 30 regions, preferably 50.

The winning estimator becomes the main \(p_v\) implementation.

---

# 26. Core ablation

On at least:

```text
ManiSkill PickCube
ManiSkill PegInsertionSide
Meta-World pick-place
FetchPickAndPlace
```

run:

```text
PPO
HERP-sigma
HERP-p
HERP
```

This is the required factorization study.

---

# 27. Estimator ablations

Only on one or two representative tasks.

Sigma:

```text
pairwise
branch
return
disagreement
```

p:

```text
occupancy
cosine
dot
fisher
hybrid
```

Do not run the full grid on all tasks.

---

# 28. Hyperparameter ablations

Only after main results work.

Probe horizon:

```text
4
8
16
```

Number of probes:

```text
2
4
8
```

Allocated fraction:

```text
0.10
0.25
0.50
```

---

# 29. Standardized restore tests

Support commands:

```bash
python scripts/check_restore.py --benchmark maniskill --env-id PickCube-v1
python scripts/check_restore.py --benchmark metaworld --env-id pick-place-v3
python scripts/check_restore.py --benchmark fetch --env-id FetchPush-v4
```

Each test must check:

1. snapshot identity,
2. same-action next-state replay,
3. reward equality,
4. termination/truncation equality,
5. episode clock restoration.

Write a JSON report.

A benchmark is paper-ready only after restore passes.

---

# 30. Server deployment cleanup

Remove hard-coded local Python paths.

Use:

```bash
PY=${PY:-python}
```

or simply:

```bash
python scripts/run_suite.py ...
```

Do not assume a WSL Conda path.

---

# 31. Device handling

Do not globally hard-code:

```text
device=cpu
sim_backend=physx_cpu
```

Put device/backend in config.

For ICRA:

- sequential CPU simulation is acceptable,
- independent seeds can run as separate worker processes,
- GPU-vectorized simulator refactor is optional.

Do not redesign the entire stack only to use H100 before the algorithm is validated.

---

# 32. Suite runner

Implement:

```bash
python scripts/run_suite.py     --benchmarks maniskill metaworld fetch     --methods ppo rnd disagreement herp_sigma herp_p herp     --seeds 0 1 2
```

Runner requirements:

1. one directory per task-method-seed,
2. skip completed runs,
3. save console logs,
4. fail visibly on crashes,
5. aggregate completed runs periodically,
6. never overwrite a changed protocol in the same output directory.

---

# 33. Reproducibility

Every run stores:

```text
config.json
metrics.csv
regions.csv
checkpoint_*.pt
complete.json
provenance.json
```

`provenance.json` should include:

- SHA-256 source hashes,
- Python version,
- PyTorch version,
- benchmark package version,
- benchmark name,
- git commit SHA if available,
- seed,
- device,
- simulator backend,
- total training interaction budget.

---

# 34. Logging

Minimum:

```text
global_env_steps
normal_steps
reference_steps
probe_steps
allocated_steps

train_return
eval_return
eval_success

policy_loss
value_loss
entropy
approx_kl
clipfrac

num_regions
num_archived_states

mean_sigma
median_sigma
max_sigma

mean_p
median_p
fraction_positive_alignment

priority_entropy
allocation_histogram

gradient_signature_norm_ref
gradient_signature_norm_region

wall_time
```

---

# 35. Tests before long runs

Required:

```text
[ ] all unit tests pass
[ ] restore tests pass on all three adapters
[ ] budget equality assertion passes
[ ] PPO learns at least one task in every benchmark family
[ ] HERP-sigma runs end-to-end
[ ] HERP-p runs end-to-end
[ ] HERP full runs end-to-end
[ ] RND runs end-to-end
[ ] Disagreement runs end-to-end
```

---

# 36. Implementation order

## Phase 1

Refactor current ManiSkill code behind `EnvAdapter` without changing behavior.

## Phase 2

Fix \(p_v\):

1. full actor gradient,
2. shared advantage scaling,
3. occupancy estimator,
4. hybrid estimator,
5. controlled PPO-delta mechanism test.

Do not launch long HERP-p runs before this.

## Phase 3

Add Meta-World adapter and exact restore test.

## Phase 4

Add Fetch adapter, dict-observation handling, goal restoration, and exact restore test.

## Phase 5

Run long benchmarks only after all smoke tests pass.

---

# 37. Main experimental matrix

Target:

```text
10 tasks
6 methods
3 seeds
```

Total:

\[
10\times6\times3=180
\]

training runs.

This is the eventual matrix; coding can finish before all runs complete.

---

# 38. Reduced matrix if compute is limited

Use:

ManiSkill:

```text
PickCube-v1
PegInsertionSide-v1
```

Meta-World:

```text
pick-place-v3
peg-insert-side-v3
```

Fetch:

```text
FetchPush-v4
FetchPickAndPlace-v4
```

Then:

```text
6 tasks
6 methods
3 seeds
= 108 runs
```

This is still sufficiently diverse for ICRA.

---

# 39. Paper success criteria

Aim for:

1. HERP improves sample efficiency over PPO on multiple tasks.
2. HERP is competitive with or better than RND and Disagreement.
3. Full HERP improves over at least one single-factor ablation consistently.
4. \(\sigma_v\) correlates with high-sample future branching.
5. the chosen \(p_v\) correlates positively with controlled PPO improvement.
6. results hold on at least two simulator families.
7. every method uses equal training interaction budget.

If gradient-based \(p_v\) remains weak but occupancy or hybrid works, use the empirically supported estimator in the main paper.
