# HERP Implementation Plan: ICRA 2027 Emergency Build

**Paper deadline:** 15 September 2026, 11:59 PM PST  
**Paper constraint:** 8 pages total, including references.  
**Primary goal:** produce a clean, defensible state-based robot-RL result before expanding the method.

The implementation should be deliberately minimal. Do not build a large framework first.

---

# 1. Frozen MVP

The submission-grade MVP is:

> **ManiSkill state-based PPO + archived simulator states + multi-step future branching score \(\sigma_v\) + reference-gradient alignment \(p_v\) + budget allocation \(q_v\propto p_v\sigma_v\).**

Everything else is optional until this works.

Use **PPO** first because:

1. ManiSkill already provides a tested official PPO baseline;
2. reset rollouts are on-policy under the current policy;
3. policy-gradient signatures for \(p_v\) are natural;
4. the code is compact enough to modify quickly.

Do not start with SAC unless PPO is clearly failing on the selected task.

---

# 2. Base codebase

## 2.1 Primary environment / PPO codebase

Use the official ManiSkill repository:

https://github.com/haosulab/ManiSkill

Relevant code:

`examples/baselines/ppo/`

The official ManiSkill PPO is adapted from CleanRL / LeanRL and supports state-based and visual RL.

Primary files to inspect:

- `examples/baselines/ppo/ppo.py`
- `examples/baselines/ppo/ppo_fast.py`
- `examples/baselines/ppo/baselines.sh`

For ICRA, use **state observations only**.

Recommended control mode for tabletop Panda tasks:

`pd_ee_delta_pose`

unless the ManiSkill task-specific PPO command uses a better tuned mode.

---

## 2.2 State restoration

ManiSkill exposes simulator state snapshots through:

```python
state = env.get_state_dict()
env.set_state_dict(state)
```

and supports resets using saved state dictionaries:

```python
obs, info = env.reset(
    options={
        "reset_to_env_states": {
            "env_states": saved_state,
            "obs": None,
        }
    }
)
```

ManiSkill also provides `CachedResetWrapper`.

This is the key infrastructure enabling HERP.

Relevant documentation:

- https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html
- https://maniskill.readthedocs.io/en/latest/user_guide/wrappers/cached_reset.html

---

# 3. Repository layout

Implement the following structure inside the HERP repository.

```text
herp/
├── train.py
├── configs/
│   ├── base.yaml
│   ├── pushcube.yaml
│   ├── pickcube.yaml
│   ├── stackcube.yaml
│   └── peginsertion.yaml
├── herp/
│   ├── archive.py
│   ├── regions.py
│   ├── probe.py
│   ├── sigma.py
│   ├── relevance.py
│   ├── allocator.py
│   ├── gradient_signature.py
│   ├── rollout_buffer.py
│   ├── eval.py
│   └── logging.py
├── baselines/
│   ├── uniform.py
│   ├── rnd.py
│   ├── disagreement.py
│   ├── plr_region.py
│   └── go_explore_region.py
├── scripts/
│   ├── smoke.sh
│   ├── main.sh
│   ├── ablation.sh
│   ├── sigma_validation.sh
│   └── p_validation.sh
└── analysis/
    ├── aggregate.py
    ├── plot_learning.py
    ├── plot_budget.py
    ├── plot_sigma_correlation.py
    └── plot_p_correlation.py
```

Do not refactor ManiSkill more than necessary.

---

# 4. Core data structures

## 4.1 Region

```python
@dataclass
class Region:
    region_id: int
    centroid: Tensor
    count: int

    snapshots: list
    last_seen_step: int
    last_probed_step: int

    sigma_raw: float = 0.0
    sigma_ema: float = 0.0

    p_raw: float = 0.0
    p_ema: float = 0.0

    priority: float = 0.0
```

Each snapshot stores:

```python
@dataclass
class Snapshot:
    env_state: dict
    obs: Tensor
    timestep: int
    episode_id: int
    return_so_far: float
```

Cap snapshots per region with reservoir sampling, e.g. `max_snapshots_per_region=8`.

---

# 5. Regionizer

For the emergency version, use normalized state features and online k-center / radius clustering.

## 5.1 Feature

```python
z = normalize(obs_state)
```

Optionally exclude:
- target-independent constant dimensions;
- raw quaternion duplication if it destabilizes Euclidean distance.

Do not learn a representation before the basic method works.

## 5.2 Assignment

```python
dist = torch.cdist(z[None], centroids)
v = argmin(dist)

if min_dist > region_radius and num_regions < max_regions:
    create_new_region(z)
else:
    assign_to(v)
    update_centroid_ema(v, z)
```

Recommended initial values:

```yaml
region_radius: 0.5
max_regions: 128
max_snapshots_per_region: 8
```

The actual radius must be tuned after observation normalization.

## 5.3 Candidate-region filter

Do not score all regions every iteration.

At each allocation round, form a candidate set from:

1. regions visited in the most recent ordinary rollout;
2. top stale regions;
3. a small uniform sample from the archive.

Start with at most 16 candidate regions per scoring round.

---

# 6. Probe rollouts

## 6.1 Probe horizon

Start with:

```yaml
probe_horizon: 8
num_action_probes: 4
num_env_repeats: 1
```

Try \(H\in\{4,8,16\}\) later.

## 6.2 Action probing

For a restored state \(x_v\), use the current policy mean plus controlled perturbation.

For continuous Gaussian PPO:

```python
mu, std = policy(obs)
eps = torch.randn_like(mu)

a_probe = clamp(
    mu + probe_scale * std * eps,
    action_low,
    action_high,
)
```

Then continue the short rollout with either:

### Option A — persistent perturbed policy

Sample current-policy actions normally for every future step.

### Option B — first-action perturbation

Perturb only the first action, then follow the current policy.

Use **Option B first**. It makes "branch point" interpretation cleaner.

Recommended:

```yaml
probe_scale: 1.0
```

Ablate `0.5, 1.0, 2.0`.

---

# 7. Future feature \(\Psi(\tau^+)\)

Use normalized state features.

```python
future_feature = weighted_mean(
    phi(s_1), ..., phi(s_H),
    weights=gamma_branch ** arange(H)
)
```

Recommended:

```yaml
gamma_branch: 0.95
```

Also log:

- endpoint feature `phi(s_H)`;
- short-horizon return;
- bootstrapped return.

These are useful for estimator ablations without recollecting data.

---

# 8. \(\sigma_v\) implementation

## 8.1 Main MVP: pairwise trajectory dispersion

```python
def pairwise_sigma(Z):
    # Z: [K, d]
    diff = Z[:, None, :] - Z[None, :, :]
    d2 = (diff ** 2).sum(-1)

    K = Z.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=Z.device)

    sigma2 = 0.5 * d2[mask].mean()
    return torch.sqrt(sigma2 + 1e-8)
```

Normalize sigma across candidate regions using robust rank or running quantiles.

Do not use raw magnitudes directly in allocation until scale is stable.

---

## 8.2 Preferred branch decomposition

When `num_env_repeats > 1`, store

```python
Z.shape == [A, M, d]
```

Compute:

```python
mean_per_action = Z.mean(dim=1)      # [A, d]
global_mean = mean_per_action.mean(0)

sigma_branch2 = (
    (mean_per_action - global_mean).pow(2).sum(-1).mean()
)

sigma_dyn2 = (
    (Z - mean_per_action[:, None, :]).pow(2).sum(-1).mean()
)

sigma2 = sigma_branch2 + lambda_dyn * sigma_dyn2
```

Default:

```yaml
lambda_dyn: 0.0
```

This is the recommended final version if the implementation is stable.

---

## 8.3 Gradient-variance oracle

For validation only, compute a gradient signature for each probe trajectory.

```python
g_k = get_policy_gradient_signature(traj_k)
sigma_grad2 = mean(||g_k - mean(g)||^2)
```

This can use only:
- actor output layer;
- last MLP layer + actor head.

It does not need full-network per-sample gradients.

---

## 8.4 Ensemble-disagreement option

Only implement if the basic \(\sigma\) experiment is weak.

Use 5 MLP forward models:

```python
f_i([phi(s), a]) -> phi(s_next)
```

Train on replayed online transitions.

Score:

```python
preds = stack([f_i(s, a) for i in ensemble])
sigma_ens = preds.var(dim=0).mean().sqrt()
```

This is both:
- a HERP estimator option;
- an implementation of a strong exploration baseline.

---

# 9. Gradient signature

Full per-region policy gradients are expensive. Use a low-dimensional signature.

## 9.1 Parameters

First attempt:

```python
signature_params = actor_head.parameters()
```

Second attempt:

```python
signature_params = last_policy_mlp_layer + actor_head
```

Do not compute gradients through the critic for \(p_v\).

## 9.2 PPO policy loss signature

For a rollout batch \(D\):

\[
L_\pi(D)
=
-\mathbb E[
\min(r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
].
\]

For scoring, preferably use the unclipped score-function gradient under the current policy to reduce dependence on old-policy ratios:

\[
L_{\mathrm{sig}}(D)
=
-
\mathbb E[
\log\pi_\theta(a_t\mid s_t)
\widehat A_t
].
\]

Then:

```python
g = autograd.grad(
    L_sig,
    signature_params,
    retain_graph=False,
    create_graph=False
)

g = torch.cat([x.flatten() for x in g])
```

Normalize before cosine similarity.

---

# 10. \(p_v\) implementation

## 10.1 Reference batch

Every `relevance_interval`, collect a small batch from **ordinary resets**, with no HERP state resetting and no exploration perturbation.

This batch is training-time reference data.

Recommended:

```yaml
reference_episodes: 16
relevance_interval: 5
```

Do not use final test seeds.

Compute:

```python
g_ref = gradient_signature(reference_batch)
```

---

## 10.2 FO cosine alignment — main

```python
p_raw = cosine_similarity(g_v, g_ref)
p_positive = relu(p_raw)
p_ema = ema(p_ema, p_positive, tau=0.9)
```

Then normalize over candidate regions:

```python
p = (p_ema + eps_p) ** alpha_p
p = p / p.sum()
```

Default:

```yaml
eps_p: 0.05
alpha_p: 1.0
```

---

## 10.3 FO dot product

Ablation:

```python
p_raw = torch.dot(g_v, g_ref)
```

Normalize by running robust scale before positive clipping.

---

## 10.4 Diagonal Fisher influence

Estimate diagonal empirical Fisher over the reference batch:

```python
F_diag = EMA(g_sample ** 2)
```

Then approximate

\[
(F+\lambda I)^{-1}g_v
\]

as:

```python
g_v_nat = g_v / (F_diag + damping)
p_raw = dot(g_ref, g_v_nat)
```

Recommended:

```yaml
fisher_damping: 1e-3
```

This should be the first "influence-function-like" second estimator.

---

## 10.5 Full IHVP — optional only

Do not block the ICRA MVP on this.

If implemented:

1. use only the signature parameter subset;
2. use Hessian-vector products;
3. solve
   \[
   (F+\lambda I)x=g_v
   \]
   by conjugate gradient;
4. score
   \[
   p_v=g_{\mathrm{ref}}^\top x.
   \]

Abort this feature if it consumes more than half a day without a stable result.

---

# 11. Allocator

## 11.1 Main score

```python
score_v = (
    (p_v + eps_p) ** alpha
    * (sigma_v + eps_sigma) ** beta
)
```

Normalize:

```python
q = score / score.sum()
```

Recommended:

```yaml
alpha: 1.0
beta: 1.0
eps_sigma: 0.05
uniform_mix: 0.10
```

Use:

```python
q = (1 - uniform_mix) * q + uniform_mix / len(q)
```

## 11.2 Budget

The most important fairness requirement:

> Every method receives exactly the same total environment-step budget.

Define a fixed split, for example:

```yaml
ordinary_interaction_fraction: 0.75
allocated_interaction_fraction: 0.25
```

If a baseline does not use state resetting, it spends the entire budget on normal rollouts.

For HERP:

```text
total steps = normal steps + probe steps + allocated reset-rollout steps
```

**Probe rollouts count toward the environment-interaction budget.**

Reusing probe rollouts for training is allowed and recommended.

---

# 12. Training integration

## 12.1 Ordinary buffer

Collect normal PPO trajectories exactly as the base implementation.

## 12.2 Reset fragments

For each allocated archived state:

1. restore state;
2. collect an \(H_{\text{train}}\)-step current-policy fragment;
3. bootstrap at the final state with critic \(V_\theta(s_H)\);
4. compute GAE within the fragment.

Recommended:

```yaml
allocated_rollout_horizon: 32
```

If task episode horizon is short, use 16.

## 12.3 Mixture

Concatenate ordinary and reset-rollout transitions before PPO optimization.

Log the source flag for every transition:

```python
source = NORMAL | PROBE | ALLOCATED
```

This enables later ablations.

---

# 13. Baselines and codebases

The baseline set should be divided into **must-run** and **nice-to-have**.

## 13.1 Must-run baselines

| Baseline | Why it is necessary | Codebase |
|---|---|---|
| PPO Uniform | Standard online RL control | ManiSkill official PPO: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo |
| HERP-\(\sigma\) | Tests branching-aware exploration alone | same HERP code; set \(p_v=1\) |
| HERP-\(p\) | Tests performance-aware training alone | same HERP code; set \(\sigma_v=1\) |
| HERP full | Proposed method | same code |
| RND-PPO | Strong classic novelty baseline | CleanRL RND: https://github.com/vwxyzjn/cleanrl ; original: https://github.com/openai/random-network-distillation |
| Disagreement | Direct competitor to dynamics-uncertainty \(\sigma\) | official: https://github.com/pathak22/exploration-by-disagreement |

These six are enough for a credible emergency submission if executed cleanly.

---

## 13.2 Strong additional baselines

### Go-Explore

Paper:
https://www.nature.com/articles/s41586-020-03157-9

Code:
https://github.com/uber-research/go-explore

Why relevant:
- stores previously reached states;
- returns to promising states;
- explores from them;
- demonstrated on a robot Fetch environment.

Do not port the entire old codebase. Implement a **Go-Explore-style region priority** using the HERP archive.

Minimal adaptation:

```python
priority = novelty_or_archive_score(region)
sample archived state by priority
roll out from it
```

This gives a fair same-backbone comparison.

### Prioritized Level Replay

Paper:
https://arxiv.org/abs/2010.03934

Code:
https://github.com/facebookresearch/level-replay

Adapt the idea from level to trajectory-space region:

```python
score_v = abs(TD_error_v) or abs(advantage_v)
priority = score_v + staleness
```

Call it **PLR-style region replay**, not original PLR, because the sampling unit has changed.

---

## 13.3 Optional baseline: ICM

Paper:
https://proceedings.mlr.press/v70/pathak17a.html

Original code:
https://github.com/pathak22/noreward-rl

Useful only if RND and Disagreement are insufficient. Do not prioritize it over the main ablations.

---

## 13.4 Cross-domain related method, not mandatory robot baseline

### GradAlign

https://arxiv.org/abs/2602.21492  
https://github.com/StigLidu/GradAlign

### InfOES

https://aclanthology.org/2026.acl-long.2206/

These are highly relevant to the \(p_v\) idea because they use validation/reference gradient alignment or influence for online RL experience/data selection.

However they are LLM-RL methods. Treat them as related work and motivation unless there is time to implement a direct "alignment-only" baseline. HERP-\(p\) already serves this role experimentally.

---

# 14. Benchmark environments

Use ManiSkill because it provides:
- robotics relevance;
- GPU parallel simulation;
- official PPO baseline;
- state snapshots/restoration;
- standardized evaluation.

## 14.1 Emergency 4-task suite

Run:

1. `PushCube-v1`
2. `PickCube-v1`
3. `StackCube-v1`
4. `PegInsertionSide-v1`

Rationale:

- PushCube: easy / relatively low branching control;
- PickCube: grasp transition creates meaningful branch points;
- StackCube: longer sequence with failure modes;
- PegInsertionSide: precision task where small action differences can create divergent futures.

If `StackCube-v1` is not in the official small PPO benchmark script, still use it if baseline PPO trains reliably. Otherwise replace it with `PushT-v1` from the official small benchmark.

---

# 15. Experimental protocol

## 15.1 Equal budget

Fix an environment-step budget per task.

All methods:
- same policy network;
- same PPO hyperparameters;
- same observation mode;
- same reward mode;
- same control mode;
- same training seeds;
- same total interaction count.

HERP's branch probes count as interaction.

## 15.2 Seeds

Submission minimum:

```text
3 seeds per method/task
```

Preferred:

```text
5 seeds per method/task
```

If time is limited:
- use 3 seeds for all main-table methods;
- add 5 seeds only for the strongest 2 tasks after the paper deadline if allowed for a later revision, not for the submitted result.

## 15.3 Evaluation

At fixed interaction checkpoints:

- 50 deterministic or low-noise evaluation episodes;
- held-out reset seeds;
- no archived-state reset;
- no exploration perturbation.

Report:

1. success rate;
2. normalized return;
3. area under learning curve;
4. environment steps to reach a fixed success threshold;
5. wall-clock time;
6. extra compute overhead.

---

# 16. Main paper table

Rows:

```text
PPO
RND
Disagreement
HERP-sigma
HERP-p
HERP
```

Columns:

```text
PushCube success
PickCube success
StackCube success
PegInsertion success
Mean normalized score
```

Also provide a sample-efficiency figure:

```text
x-axis: environment interactions
y-axis: success rate
```

Use 2 representative tasks in the main plot due to the ICRA 8-page limit.

---

# 17. Mechanism experiment A: does \(\sigma_v\) detect branch points?

This is mandatory.

## Procedure

At 2--3 policy checkpoints:

1. sample 50--100 archived regions;
2. estimate \(\hat\sigma_v\) with \(K=4\) futures;
3. collect an expensive oracle with 64 futures from the same saved state;
4. compute oracle trajectory variance;
5. compute gradient-variance oracle on the policy head if affordable.

Report:

- Spearman correlation:
  \[
  \rho(\hat\sigma_v,\sigma_v^{64})
  \]
- optional:
  \[
  \rho(\hat\sigma_v,\sigma_{g,v})
  \]
- top-5 and bottom-5 region visualizations / trajectory plots.

A good paper figure is:

```text
left: state snapshots with low sigma
right: state snapshots with high sigma
plus short future trajectories
```

---

# 18. Mechanism experiment B: does \(p_v\) predict useful updates?

This is mandatory for the "performance-aware" claim.

## Procedure

At a fixed checkpoint:

1. sample 30--50 regions;
2. compute:
   - occupancy relevance;
   - cosine alignment;
   - dot alignment;
   - diagonal-Fisher influence;
3. clone current policy;
4. for each region, make one small policy update using only region data;
5. evaluate the change in reference return / reference policy loss.

Define ground truth:

\[
\Delta_v
=
J_{\mathrm{ref}}(\theta_v^+)
-
J_{\mathrm{ref}}(\theta).
\]

Report Spearman correlation:

\[
\rho(p_v,\Delta_v).
\]

This figure can directly justify the chosen \(p_v\) estimator.

---

# 19. Core ablations

Run on at least two representative tasks.

## 19.1 Score decomposition

```text
Uniform
sigma only
p only
p * sigma
```

This is the single most important ablation.

## 19.2 Sigma estimator

```text
return variance
pairwise trajectory dispersion
branch decomposition
ensemble disagreement
```

If compute is limited, compare only:
- pairwise;
- branch decomposition;
- ensemble.

## 19.3 p estimator

```text
occupancy
FO cosine
FO dot
diag Fisher
```

## 19.4 Probe horizon

```text
H = 4, 8, 16
```

## 19.5 Number of probe futures

```text
K = 2, 4, 8
```

## 19.6 Allocated interaction fraction

```text
0.10
0.25
0.50
```

---

# 20. Critical anti-confounds

The following mistakes will invalidate the main result.

## 20.1 Do not use final test data to compute \(p_v\)

Training-time reference rollouts and test rollouts must use different seeds / instances.

## 20.2 Count probe interactions

Do not give HERP more simulator interactions than baselines.

## 20.3 Keep PPO identical

Do not silently retune PPO only for HERP.

## 20.4 Reset only to previously reached states

Do not synthesize impossible simulator states for the main method.

## 20.5 Separate branchiness from random noise

If a stochastic environment produces high \(\sigma\) solely because of uncontrollable noise, test the branch decomposition and reduce \(\lambda_{\text{dyn}}\).

---

# 21. Logging

Every run should log at minimum:

```text
global_env_steps
wall_time
train_return
eval_return
eval_success

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

normal_steps
probe_steps
allocated_steps

policy_loss
value_loss
entropy
approx_kl
clipfrac

gradient_signature_norm_ref
gradient_signature_norm_region
```

For each region score event, dump a compact row:

```text
step, env_id, region_id, sigma, p, priority,
visit_count, last_seen, last_probed
```

Save to Parquet or CSV.

---

# 22. Smoke-test checklist

Before launching expensive experiments:

### Environment
- [ ] ManiSkill PPO reproduces learning on PushCube.
- [ ] `get_state_dict` snapshot restores correctly.
- [ ] same action from same restored state approximately reproduces the same next state in deterministic mode.

### Archive
- [ ] regions are created;
- [ ] region count does not explode;
- [ ] snapshots are bounded by reservoir size.

### Sigma
- [ ] identical futures give near-zero score;
- [ ] action perturbations increase score at visibly sensitive states;
- [ ] score scale does not explode.

### p
- [ ] `g_ref` is non-zero;
- [ ] region cosine similarities span both low and high values;
- [ ] p is recomputed after policy changes.

### Budget
- [ ] normal + probe + allocated steps exactly equal configured total.

### PPO
- [ ] reset fragments produce finite GAE;
- [ ] no NaNs after mixing ordinary/reset data.

---

# 23. Run plan

Use a config interface such as:

```bash
python train.py \
  --env-id PickCube-v1 \
  --method herp \
  --seed 0 \
  --total-timesteps 3000000 \
  --probe-horizon 8 \
  --num-probes 4 \
  --allocated-frac 0.25 \
  --sigma-estimator branch \
  --p-estimator cosine
```

Baselines:

```bash
python train.py --env-id PickCube-v1 --method ppo --seed 0
python train.py --env-id PickCube-v1 --method rnd --seed 0
python train.py --env-id PickCube-v1 --method disagreement --seed 0
python train.py --env-id PickCube-v1 --method herp_sigma --seed 0
python train.py --env-id PickCube-v1 --method herp_p --seed 0
python train.py --env-id PickCube-v1 --method herp --seed 0
```

---

# 24. Emergency schedule to ICRA

The official ICRA 2027 submission deadline is **15 September 2026, 11:59 PM PST** and the call states there is no planned extension.

## 9 Sep — freeze + infrastructure

Must finish:
- ManiSkill PPO running;
- state save / reset working;
- archive + simple regions;
- HERP allocator stub;
- uniform method reproducing baseline.

Do not work on full influence functions today.

## 10 Sep — sigma + full loop

Must finish:
- short restored-state probe rollout;
- pairwise trajectory \(\sigma\);
- cosine \(p\);
- full `p * sigma` allocation;
- one PushCube / PickCube end-to-end run.

Decision at end of day:
- if method is unstable, freeze pairwise sigma + cosine p and stop adding sophistication.

## 11 Sep — main experiments

Launch:
- PPO;
- RND;
- Disagreement;
- sigma-only;
- p-only;
- HERP.

Start with:
- PickCube;
- PegInsertion;
- PushCube.

Add StackCube only when compute permits.

## 12 Sep — mechanism validation

Run:
- sigma oracle correlation;
- p vs actual-update correlation;
- visualization of selected states.

These experiments are more valuable than a large number of weak extra baselines.

## 13 Sep — essential ablations + aggregate

Run:
- score decomposition;
- H/K sensitivity on 1--2 tasks;
- compile main table and learning curves.

Freeze experiments by night unless a single missing run is critical.

## 14 Sep — paper writing

Produce:
- method figure;
- main table;
- 2 learning curves;
- sigma mechanism figure;
- p correlation figure.

Write the 8-page ICRA manuscript around existing results, not around planned results.

## 15 Sep — final checks

Only:
- missing seeds if short;
- reproducibility checks;
- typo / formatting;
- anonymization;
- PaperPlaza upload.

Do not introduce a new algorithmic component on submission day.

---

# 25. Priority if compute/time collapses

If only enough time for a minimal paper, run:

## Tasks

```text
PickCube-v1
PegInsertionSide-v1
PushCube-v1
```

## Methods

```text
PPO
RND
Disagreement
HERP-sigma
HERP-p
HERP
```

## Seeds

```text
3
```

## Mechanism studies

```text
sigma oracle correlation on PickCube
p correlation on PickCube
score decomposition on PickCube + PegInsertion
```

This is preferable to running 10 baselines with poor controls.

---

# 26. Success criteria before paper freeze

The proposed method should satisfy at least three of the following before submission:

1. HERP improves final success over PPO on at least 2 non-trivial tasks.
2. HERP improves learning-curve AUC / steps-to-threshold under equal interaction budget.
3. Full \(p\sigma\) consistently beats either \(\sigma\)-only or \(p\)-only.
4. Estimated \(\sigma_v\) correlates with high-sample future branching.
5. Estimated \(p_v\) correlates with actual short-update reference improvement.
6. Selected high-priority regions are qualitatively interpretable as task-relevant branching points.

If the full method does not beat its single-factor ablations, do not claim a joint-allocation contribution yet. Reframe around whichever estimator is actually supported by the evidence.

---

# 27. Main paper story if results work

The paper can be written as:

> Existing exploration methods largely prioritize novelty or uncertainty, while replay/curriculum methods prioritize learning potential. In robot online RL, these signals should be coupled at the level of trajectory-space regions. HERP restores previously reached states and allocates a fixed interaction budget according to two online estimates: multi-step future branching \(\sigma_v\) and reference-objective gradient relevance \(p_v\). A stratified-sampling argument gives the allocation structure \(n_v\propto p_v\sigma_v\). Experiments show that the two quantities predict branching and useful policy updates respectively, and their joint allocation improves sample efficiency under equal simulator budgets.

That is the version to target first.
