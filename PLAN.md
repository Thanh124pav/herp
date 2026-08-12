# PLAN.md — MVP for Receiver-Aware Experience Routing in Robot Policy Populations

## 0. Goal

Build the smallest end-to-end implementation that can test the central research hypothesis:

> A population of independently learning robot policies can discover complementary functional experiences, and routing those experiences from strong donors to weak receivers can improve learning over independent training, indiscriminate shared replay, and global-priority sharing.

The MVP must **not** introduce a learned visual/trajectory embedding model, VLA/VLM training, learned skill segmentation, or a full RL coordinator. The main scientific contribution to preserve is the **experience-level donor–receiver routing problem**, not representation learning.

The initial pipeline is:

```text
Parallel online RL
    ↓
Trajectory segmentation
    ↓
State-transition experience grouping
    ↓
Dynamic policy × experience competence matrix
    ↓
Minimax-style receiver deficit / demand
    ↓
Unbalanced OT donor–receiver assignment
    ↓
Direct replay of routed experience
    ↓
Policy updates
```

Core interpretation:

```text
Experience discovery:  What functional experiences exist?
Competence matrix:      Who can execute each experience?
Minimax deficit:        Who is missing what?
Optimal transport:      Who should teach what to whom?
```

---

# 1. Scope Lock

## 1.1. In scope for the MVP

- One robot embodiment.
- State-based / low-dimensional observations.
- Continuous-control manipulation.
- Off-policy actor–critic RL, preferably SAC.
- `N = 4` same-architecture policies as the default population size.
- Policies differ by:
  - initialization seed;
  - exploration randomness;
  - initial object configuration;
  - mild controlled environment/dynamics perturbation.
- Separate local replay buffer for each policy.
- Simple state-based trajectory segmentation.
- Dynamic experience vocabulary.
- State-transition-based experience equivalence.
- Frequency-in-successful-trajectories filter for valid experiences.
- Online policy × experience competence matrix.
- Minimax-inspired receiver deficit matrix.
- Greedy routing baseline.
- Unbalanced OT routing.
- Direct replay as the only transfer mechanism.
- Matched total environment-interaction budgets across primary comparisons.

## 1.2. Explicitly out of scope

Do **not** implement these unless the MVP fails specifically because one is necessary:

- VLA policies.
- VLM-based semantic embeddings.
- Image-only observations.
- Learned trajectory encoder `f`.
- Contrastive skill representation learning.
- Learned semantic segmentation.
- Language annotations.
- Skill decoder / skill VAE.
- Meta-gradient routing.
- Learned RL coordinator.
- Multiple transfer modes.
- Cross-embodiment transfer.
- Real-robot deployment.
- Joint-action cooperative MARL.
- Parameter transfer / whole-policy distillation as part of the proposed method.

These can later become baselines, ablations, or future work.

---

# 2. Repositories to Reuse or Read

The codebase should **not** be a direct fork of one paper. Create a clean project and keep prior repositories as read-only references under `external/` or as git submodules.

## 2.1. Primary implementation reference: QMP

Repository:

```bash
git clone https://github.com/clvrai/qmp external/qmp
```

Use QMP as the main reference for:

- multiple off-policy policies;
- separate policy and Q-function objects;
- policy-population initialization;
- continuous-control RL structure;
- behavior-sharing experiment organization;
- Meta-World / MuJoCo-style experiments;
- separated-policy baselines;
- receiver-side Q evaluation baseline.

Relevant QMP structure to inspect first:

```text
qmp/
├── learning/
├── environments/
├── experiment_utils.py
├── garage_experiments.py
└── run.py
```

Important: do **not** inherit QMP's multi-task assumption as the proposed setting. In our MVP, all policies solve the same task/task family but have different learning histories and mild environment perturbations.

Also do not depend on QMP's old environment stack if it creates version conflicts. Port the useful learning logic into the new codebase.

### QMP code pieces to understand

1. policy/Q initialization;
2. separated SAC baseline;
3. data-sharing baseline;
4. mixture-policy wrapper;
5. receiver Q-function evaluation of candidate actions.

QMP-style behavior selection is a **baseline**, not our routing mechanism.

---

## 2.2. Segmentation reference: EXTRACT

Repository:

```bash
git clone https://github.com/jesbu1/extract external/extract
```

Use EXTRACT only as a conceptual/code reference for:

```text
per-timestep features
    → clustering
    → temporal label smoothing
    → contiguous chunks
```

Inspect especially:

```text
vlm_cluster_dataset.py
view_frames_from_cluster.py
```

EXTRACT uses K-means and median filtering for its skill-clustering pipeline.

Do **not** inherit:

- VLM features;
- VAE skill model;
- skill prior;
- skill decoder;
- staged skill-policy training.

Our MVP replaces the semantic VLM representation with normalized task-relevant state-transition features.

---

## 2.3. Selective-sharing baseline: SUPER

Repository:

```bash
git clone https://github.com/mgerstgrasser/super external/super
```

SUPER is a DQN/RLlib implementation and should **not** be used as the backbone.

Use it to understand and reproduce the algorithmic baseline:

```text
share a limited subset of high-TD-error experience
```

In our experiments, implement a **SUPER-style SAC adaptation inside our own backbone** so that environment, policy architecture, optimizer, and replay budget are matched.

Do not compare our SAC implementation numerically against SUPER's original DQN experiments.

---

## 2.4. Shared-experience reference: SEAC

Repository:

```bash
git clone https://github.com/uoe-agents/seac external/seac
```

SEAC is useful as prior art for learning from other agents' experience.

Its official experiments target cooperative multi-agent environments such as LBF and RWARE, so it is not a natural backbone for this project.

Use SEAC for:

- understanding cross-policy experience reuse;
- an optional importance-weighted sharing baseline if it can be adapted cleanly.

SEAC is **not required for the first MVP milestone**.

---

## 2.5. Environment: Meta-World

Repository:

```bash
git clone https://github.com/Farama-Foundation/Metaworld external/metaworld
```

Use the current Farama Meta-World implementation rather than trying to keep an old QMP/Garage environment stack alive.

Why Meta-World first:

- same robot morphology across many manipulation tasks;
- low-dimensional state access;
- fast simulation compared with VLA training;
- interpretable functional phases;
- easy success metric;
- compatible with the scientific question.

Start with one task before multi-task evaluation.

Suggested first task family:

```text
pick-place
push
reach
```

The first smoke test should use only one task.

---

## 2.6. Second environment after the MVP: ManiSkill

Repository:

```bash
git clone https://github.com/mani-skill/ManiSkill external/maniskill
```

Do not add ManiSkill until the full method works in Meta-World.

Use ManiSkill only for:

- second-suite validation;
- stronger dynamics/randomization stress tests;
- checking that conclusions are not Meta-World-specific.

---

## 2.7. OT solver: Python Optimal Transport (POT)

Repository:

```bash
git clone https://github.com/PythonOT/POT external/POT
```

Install as a normal dependency:

```bash
pip install POT
```

Use POT for:

- unbalanced Sinkhorn OT;
- optionally partial Wasserstein OT.

Preferred MVP solver:

```python
ot.unbalanced.sinkhorn_unbalanced(...)
```

Reason for using unbalanced/partial OT:

- not every donor experience should be transferred;
- not every receiver deficit must be filled;
- digital experience is not a physical conserved resource;
- unused supply/demand should be allowed.

---

# 3. Recommended Project Structure

```text
experience-routing/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── env/
│   ├── policy/
│   ├── population/
│   ├── experience/
│   └── routing/
│
├── external/                  # read-only prior repositories
│
├── src/
│   └── experience_routing/
│       ├── envs/
│       │   ├── metaworld_env.py
│       │   └── perturbations.py
│       │
│       ├── rl/
│       │   ├── sac.py
│       │   ├── actor.py
│       │   ├── critic.py
│       │   └── replay_buffer.py
│       │
│       ├── population/
│       │   ├── worker.py
│       │   ├── population.py
│       │   └── rollout_manager.py
│       │
│       ├── experience/
│       │   ├── trajectory.py
│       │   ├── segmenter.py
│       │   ├── grouper.py
│       │   ├── vocabulary.py
│       │   ├── validity.py
│       │   └── bank.py
│       │
│       ├── competence/
│       │   ├── tracker.py
│       │   ├── opportunity.py
│       │   └── deficit.py
│       │
│       ├── routing/
│       │   ├── base.py
│       │   ├── no_share.py
│       │   ├── share_all.py
│       │   ├── random.py
│       │   ├── td_priority.py
│       │   ├── qmp_style.py
│       │   ├── greedy.py
│       │   └── uot.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── routing_metrics.py
│       │   └── budget.py
│       │
│       └── utils/
│
├── scripts/
│   ├── train_single.py
│   ├── train_population.py
│   ├── inspect_chunks.py
│   ├── inspect_experiences.py
│   ├── inspect_competence.py
│   ├── run_baseline.py
│   └── run_full.py
│
├── tests/
│   ├── test_segmenter.py
│   ├── test_grouper.py
│   ├── test_vocabulary.py
│   ├── test_competence.py
│   ├── test_deficit.py
│   └── test_ot_router.py
│
└── outputs/
```

Keep the proposed method modular enough that routing can be changed without touching SAC or the environment.

---

# 4. Core Data Structures

## 4.1. Trajectory

```python
@dataclass
class Trajectory:
    policy_id: int
    episode_id: int
    states: np.ndarray       # [T+1, state_dim]
    actions: np.ndarray      # [T, action_dim]
    rewards: np.ndarray      # [T]
    dones: np.ndarray        # [T]
    success: bool
    env_context: dict
```

## 4.2. Chunk

```python
@dataclass
class Chunk:
    chunk_id: int
    policy_id: int
    episode_id: int
    start_t: int
    end_t: int

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    pre_state: np.ndarray
    post_state: np.ndarray
    effect: np.ndarray

    success_episode: bool
    experience_id: int | None = None
```

Use:

```python
effect = post_state - pre_state
```

on normalized task-relevant state dimensions.

## 4.3. Experience prototype

```python
@dataclass
class Experience:
    experience_id: int

    precondition_center: np.ndarray
    effect_center: np.ndarray

    n_chunks: int
    n_success_chunks: int
    n_success_episodes: int

    valid: bool

    donor_chunk_ids: dict[int, list[int]]
```

Do not add a learned latent vector to this object in the MVP.

## 4.4. Competence matrix

At routing round `t`:

```python
competence.shape == [num_policies, num_experiences]
```

Store both estimate and evidence count:

```python
competence_mean[i, e]
competence_trials[i, e]
competence_successes[i, e]
```

## 4.5. Routing plan

```python
@dataclass
class Route:
    donor_id: int
    receiver_id: int
    experience_id: int
    chunk_ids: list[int]
    mass: float
```

The final UOT plan is converted into a finite number of actual chunks before adding data to receiver replay.

---

# 5. Module A — Policy Population

## Objective

Run `N` policies that:

- share architecture;
- solve the same task;
- have different initial seeds / exploration history;
- collect data independently;
- have separate replay buffers.

Default:

```yaml
population_size: 4
```

Do not share any data initially.

## Required first result

Before implementing experience discovery, verify:

```text
N independent SAC policies
→ all learn to some degree
→ trajectories are not identical
→ success curves differ across seeds
```

If plain SAC does not learn the chosen task, stop and fix the RL backbone before adding routing.

---

# 6. Module B — Naive EXTRACT-Style Trajectory Segmentation

The segmentation module is intentionally **not** a research contribution.

## 6.1. Per-step state-transition feature

Select task-relevant state dimensions from Meta-World.

Example conceptual vector:

```text
end-effector position
gripper state
object position
object-to-goal relative position
contact/grasp indicator if available
```

For each timestep:

```python
g_t = [
    delta_ee_position,
    delta_object_position,
    delta_gripper,
    delta_object_goal_distance,
    contact_change,
]
```

Normalize each feature dimension over the replay dataset.

## 6.2. EXTRACT-style temporal labels

At a refresh interval:

1. collect per-step features `g_t`;
2. run K-means with small `K_seg`;
3. assign a cluster label to every timestep;
4. median-filter labels over time;
5. split when the smoothed label changes;
6. merge chunks shorter than `min_chunk_len`.

Pseudo-code:

```python
labels = kmeans.predict(step_features)
labels = median_filter(labels, size=median_window)

boundaries = np.where(labels[1:] != labels[:-1])[0] + 1
chunks = split_trajectory(trajectory, boundaries)
chunks = merge_short_chunks(chunks, min_chunk_len)
```

Start with:

```yaml
K_seg: 6
median_window: 7
min_chunk_len: 4
```

These are starting values, not fixed scientific claims.

## 6.3. Required diagnostic

Save plots for 100 trajectories:

```text
time
reward
success/failure
cluster label
chunk boundaries
selected state deltas
```

The chunker passes the MVP sanity check if:

- it does not split at nearly every timestep;
- it does not return only one chunk per trajectory;
- boundaries are temporally stable;
- chunks correspond to visibly different state-transition regimes often enough to be usable.

No semantic labels such as "grasp" are required.

---

# 7. Module C — State-Transition Experience Grouping

This is separate from segmentation.

Segmentation answers:

```text
Where does a candidate experience start/end?
```

Grouping answers:

```text
Which chunks instantiate the same functional experience?
```

## 7.1. Experience descriptor

For chunk `κ = τ[i:j]`:

```text
precondition = normalized task-relevant state at i
effect       = normalized state[j] - normalized state[i]
```

Descriptor:

```python
x_kappa = (precondition, effect)
```

Do not learn an encoder.

## 7.2. Experience equivalence

Two chunks are considered instances of the same experience if they have:

```text
similar preconditions
+
similar effects
```

Distance:

```python
d_pre = ||pre_a - pre_b||_2
d_eff = ||effect_a - effect_b||_2

d = alpha_pre * d_pre + alpha_eff * d_eff
```

Default:

```yaml
alpha_pre: 0.5
alpha_eff: 0.5
```

An online chunk is assigned to the nearest experience prototype if:

```python
d_min <= eps_experience
```

Otherwise create a new experience cluster.

## 7.3. Online vocabulary update

Pseudo-code:

```python
def assign_experience(chunk):
    if len(vocabulary) == 0:
        return create_new_experience(chunk)

    distances = [
        experience_distance(chunk, e)
        for e in vocabulary
    ]

    e_star = argmin(distances)

    if distances[e_star] <= eps_experience:
        update_prototype(e_star, chunk)
        return e_star.id
    else:
        return create_new_experience(chunk)
```

Use incremental means for prototype updates.

Periodically merge two experiences if their prototype distance is below:

```yaml
eps_merge < eps_experience
```

This prevents vocabulary fragmentation.

---

# 8. Module D — Valid / Useful Experience Filter

The MVP follows the simple hypothesis:

> A functional experience is more likely to be useful if it repeatedly occurs in successful trajectories.

For experience `e`:

```python
success_support[e] =
    num_successful_episodes_containing_e
    / num_successful_episodes
```

Require both:

```python
n_successful_episodes_containing_e >= min_success_support
success_support[e] >= success_support_threshold
```

Example starting values:

```yaml
min_success_support: 5
success_support_threshold: 0.05
```

Do not call this score receiver-specific utility.

Use terminology:

```text
experience validity / success support
```

Optional diagnostic only:

```python
success_lift[e] =
    P(e | success) - P(e | failure)
```

This diagnostic can reveal frequent but non-discriminative chunks, but it is not required in the first routing implementation.

---

# 9. Module E — Competence Estimation

This module turns dynamic experiences into a dynamic performance vector for every policy.

For `N` policies and `K_t` valid experiences:

```text
C_t ∈ R^(N × K_t)
```

The vocabulary can grow over time.

## 9.1. Avoid using raw chunk frequency as competence

Define each experience `e` by:

```text
precondition region
+
target effect / postcondition region
```

An **opportunity** occurs when policy `i` reaches a state close to the precondition prototype of `e`.

The policy successfully executes experience `e` if, within a horizon `H_e`, it realizes the corresponding effect/postcondition.

Formally:

```text
opportunity:
    d_pre(s_t, e.precondition) <= eps_pre

success:
    d_eff(s_{t+h} - s_t, e.effect) <= eps_effect
    for some h <= H_e
```

Then estimate:

```python
c[i, e] = successes[i, e] / opportunities[i, e]
```

Use Beta smoothing:

```python
c[i, e] = (successes[i, e] + alpha)
          / (opportunities[i, e] + alpha + beta)
```

Default:

```yaml
alpha: 1.0
beta: 1.0
```

If opportunities are too few, mark the competence estimate as uncertain rather than forcing it to zero.

## 9.2. Required output

Log a heatmap:

```text
rows    = policies
columns = discovered experiences
value   = competence
```

The method is only interesting if some experiences show complementary structure, e.g.:

```text
Policy A > Policy B on e1
Policy B > Policy A on e2
```

If one policy dominates every experience, routing may collapse to teacher–student transfer and the population hypothesis should be reconsidered.

---

# 10. Module F — Minimax-Inspired Deficit / Receiver Demand

For each valid experience:

```python
frontier[e] = max_i competence[i, e]
```

Receiver deficit:

```python
deficit[i, e] = max(
    0.0,
    frontier[e] - competence[i, e]
)
```

This is the operational output of the minimax stage.

Interpretation:

```text
large deficit[i, e]
= policy i is far behind the best capability already present in the population
```

The minimax stage does **not** choose the donor.

It answers only:

```text
Who is missing what?
```

Normalize deficits into OT demand:

```python
demand[i, e] = deficit[i, e] ** demand_power
```

Optional temperature/softmax can be added later.

Required visualization:

```text
receiver × experience deficit heatmap
```

---

# 11. Module G — Donor Supply

For every donor `d` and experience `e`, supply depends on:

1. donor competence on `e`;
2. number of successful stored chunks for `e`;
3. experience validity.

Simple MVP:

```python
supply[d, e] =
    competence[d, e]
    * min(
        len(successful_chunks[d, e]),
        max_chunks_per_experience
      )
```

Exclude:

```text
d == r
```

during routing.

For each `(d, e)`, keep an experience bank containing successful exemplar chunks.

When the router requests `m` units from `(d, e)`, choose exemplar chunks by:

1. successful episode only;
2. closest distance to the experience prototype;
3. optionally high return-to-go as a tie breaker.

---

# 12. Module H — Greedy Routing Baseline

Before implementing OT, implement a deterministic greedy router.

For every receiver deficit `(r, e)`:

```python
d_star = argmax_d competence[d, e]
```

Route one or more chunks from:

```text
(d_star, e) → (r, e)
```

subject to routing budget.

This baseline is essential.

If UOT cannot beat greedy best-donor matching, OT may not be necessary.

---

# 13. Module I — Unbalanced Optimal Transport Router

## 13.1. Source and target nodes

Source nodes:

```text
(donor policy d, experience e)
```

Target nodes:

```text
(receiver policy r, experience e)
```

For the MVP, only route **the same discovered experience**:

```text
e_source == e_target
```

Do not solve semantic cross-experience matching yet.

## 13.2. Supply

```python
a[(d, e)] = normalized_supply[d, e]
```

## 13.3. Demand

```python
b[(r, e)] = normalized_deficit[r, e]
```

## 13.4. Cost matrix

Keep the first cost deliberately simple.

For valid same-experience pairs:

```python
cost[(d,e), (r,e)] =
    lambda_quality * (1 - competence[d,e])
    + lambda_redundancy * competence[r,e]
```

Invalid cross-experience pairs receive a very large cost:

```python
cost[(d,e1), (r,e2)] = BIG_COST
if e1 != e2
```

Self-transfer is prohibited:

```python
cost[(d,e), (d,e)] = BIG_COST
```

Do **not** add a learned utility model in the MVP.

## 13.5. Solve

Use POT:

```python
gamma = ot.unbalanced.sinkhorn_unbalanced(
    a,
    b,
    cost,
    reg=ot_entropy_reg,
    reg_m=ot_marginal_reg,
)
```

Convert transport mass into a finite routing batch.

Example:

```python
num_chunks = round(
    routing_budget * gamma[src, dst] / gamma.sum()
)
```

## 13.6. Why unbalanced OT

The router must be able to:

- ignore low-quality donor supply;
- leave small deficits unfilled;
- not force all experience to be shared;
- respect a finite communication/replay budget.

---

# 14. Module J — Receiver Update

Direct replay only.

Each policy has:

```text
local_buffer
routed_buffer
```

Training mini-batch:

```python
batch_local  = local_buffer.sample(B_local)
batch_route  = routed_buffer.sample(B_route)

batch = concat(batch_local, batch_route)
```

Control routed-data fraction:

```yaml
route_batch_fraction: 0.25
```

Start with no importance correction.

If routed data destabilize SAC, add:

1. lower route fraction;
2. age filtering;
3. policy-mismatch rejection;
4. importance correction only after those simpler fixes.

---

# 15. Online Training Loop

Pseudo-code:

```python
initialize N independent SAC policies
initialize local replay buffers
initialize empty experience vocabulary
initialize experience banks
initialize competence tracker

for env_step in range(total_steps):

    # 1. collect experience independently
    for policy_i in population:
        transition = policy_i.step_env()
        local_buffer[i].add(transition)
        trajectory_store[i].append(transition)

    # 2. standard local SAC updates
    for policy_i in population:
        policy_i.update(
            local_buffer[i],
            routed_buffer[i],
        )

    # 3. when episodes finish, process trajectories
    for finished_trajectory in finished_trajectories:
        chunks = segmenter.segment(finished_trajectory)

        for chunk in chunks:
            e = experience_vocabulary.assign(chunk)
            experience_bank.add(e, chunk)

    # 4. periodically refresh valid experiences
    if env_step % experience_refresh_interval == 0:
        validity.update(experience_vocabulary, trajectory_store)

    # 5. update competence online
    competence_tracker.observe_population_rollouts(...)

    # 6. periodically route
    if env_step % routing_interval == 0:

        C = competence_tracker.matrix()
        D = compute_deficits(C)
        S = compute_supply(C, experience_bank)

        routes = router.solve(
            supply=S,
            demand=D,
            budget=routing_budget,
        )

        for route in routes:
            chunks = experience_bank.sample(route)
            routed_buffer[route.receiver].add_chunks(chunks)

    # 7. evaluate/checkpoint
    if env_step % eval_interval == 0:
        evaluate_all_policies()
        log_experience_vocabulary()
        log_competence_heatmap()
        log_deficit_heatmap()
        log_routing_matrix()
```

---

# 16. Implementation Milestones

## Milestone 0 — Environment + single SAC

Deliverable:

```text
single SAC policy solves one Meta-World task
```

Required tests:

- deterministic evaluation works;
- seed reproducibility works;
- success metric is logged;
- replay buffer works.

Do not proceed until this is stable.

## Milestone 1 — Independent population

Deliverable:

```text
N=4 independent policies
N local replay buffers
matched rollout accounting
```

Required outputs:

- per-policy return curves;
- per-policy success curves;
- total population interactions.

## Milestone 2 — Shared replay baselines

Implement:

```text
share-all
random cross-policy sharing
```

This creates the first experience-sharing comparison before any chunking exists.

## Milestone 3 — Chunking diagnostics

Implement:

```text
state-transition features
K-means labels
median smoothing
chunk extraction
```

Required human-inspection artifact:

```text
100 trajectories with boundary plots
```

No routing yet.

## Milestone 4 — Dynamic experience vocabulary

Implement:

```text
precondition/effect descriptor
online threshold clustering
prototype update
cluster merge
success support filter
```

Required outputs:

- `K_t` over training time;
- cluster support histogram;
- examples of chunks within each experience;
- nearest-cluster distances.

## Milestone 5 — Competence matrix

Implement opportunity/success tracking.

Required outputs:

```text
policy × experience competence heatmap
policy × experience observation-count heatmap
```

Stop and inspect before routing.

A useful MVP requires at least some complementary cells.

## Milestone 6 — Greedy best-donor routing

Implement:

```text
deficit matrix
best donor per receiver-experience deficit
direct replay
routing budget
```

This is the first proposed-routing prototype.

If greedy routing does not improve anything over shared replay, debug here before adding OT.

## Milestone 7 — UOT routing

Implement POT-based UOT.

Required comparisons:

```text
Greedy
vs
UOT
```

Required visualization:

```text
donor-experience → receiver-experience transport matrix
```

## Milestone 8 — Priority-sharing baselines

Implement in the same SAC backbone:

```text
SUPER-style top-TD-error sharing
QMP-style receiver-Q behavior selection
```

SEAC-style sharing is optional at this stage.

## Milestone 9 — Full MVP evaluation

Run full comparison with matched budgets.

Only after this milestone should learned utility prediction, better chunking, or a second environment be considered.

---

# 17. Baselines Required for the First Paper/MVP

## B0 — Single SAC

Purpose:

```text
Does a population help at all?
```

Primary comparison should match total environment interactions.

Also report a supplementary equal-per-policy-budget comparison.

## B1 — Independent population

```text
N policies
no sharing
```

This is the most important baseline for isolating routing gains from parallel exploration.

## B2 — Shared Replay / Share-All

All policies can learn from all experience.

Purpose:

```text
Is selective routing better than unrestricted experience aggregation?
```

## B3 — Random Routing

Use the exact same routing bandwidth as the proposed method, but select random donor chunks/receivers.

Purpose:

```text
Is the routing decision meaningful, or is any cross-policy data sufficient?
```

## B4 — Global TD-Priority / SUPER-Style Sharing

Adapt SUPER's central idea to SAC:

```text
select top-q TD-error transitions/chunks globally
share them with other policies
```

Use identical:

- SAC architecture;
- replay budget;
- population size;
- number of routed samples.

Purpose:

```text
receiver-specific deficit routing
vs
global experience importance
```

Repository reference:

```text
mgerstgrasser/super
```

## B5 — QMP-Style Receiver-Q Behavior Sharing

Adapt QMP's core receiver-side selection:

At a receiver state:

1. each population policy proposes an action;
2. receiver's critic evaluates candidate actions;
3. execute/select the highest-Q candidate.

Purpose:

```text
experience routing
vs
receiver-side whole-policy/action behavior selection
```

Repository reference:

```text
clvrai/qmp
```

This is conceptually close and should be treated as an important baseline.

## B6 — Greedy Best-Donor Experience Routing

Use the same discovered experiences and competence matrix as the full method.

For each `(receiver, experience)` deficit:

```text
choose highest-competence donor
```

No OT.

Purpose:

```text
Does optimal transport add anything beyond obvious donor matching?
```

## B7 — Full Method: UOT Experience Routing

```text
state-based experience discovery
+
dynamic competence matrix
+
minimax deficit demand
+
UOT routing
+
direct replay
```

---

# 18. Optional Baselines After MVP

Only add these after B0–B7 are stable.

## SEAC-style shared actor–critic experience

Reference:

```text
uoe-agents/seac
```

Add only if a fair continuous-control adaptation is straightforward.

## PBT-style policy copying

Use as a whole-policy-transfer contrast:

```text
copy/exploit globally stronger policies
```

Useful if the paper needs to show that local experience transfer is better than simply copying the best policy.

## EXTRACT-style learned skill abstraction

Not required.

Could become an ablation:

```text
naive state-transition experiences
vs
stronger learned semantic skill extraction
```

Only do this if reviewers/advisor specifically demand it.

---

# 19. Experiments

## 19.1. Experiment 1 — Does complementary experience emerge?

Setup:

```text
1 Meta-World task
N=4 independent SAC policies
different seeds
mild environment perturbations
no sharing
```

Measure:

- experience vocabulary size `K_t`;
- per-policy experience coverage;
- competence matrix;
- fraction of experiences for which the best policy differs;
- average frontier gap.

Key diagnostic:

```python
best_policy_per_experience = argmax_i C[i, e]
```

If the same policy is best on almost every experience, the local-comparative-advantage hypothesis is weak.

## 19.2. Experiment 2 — Does chunk grouping look functionally meaningful?

For each of the largest experience clusters:

- render 10 representative chunks;
- show pre-state;
- show post-state/effect;
- show successful/failure episode source;
- manually inspect whether members realize approximately the same functional state transition.

No claim of semantic-optimal clustering is required.

This is a sanity check, not a headline result.

## 19.3. Experiment 3 — Does deficit identify useful receiver needs?

For selected `(receiver, experience)` pairs:

1. checkpoint receiver;
2. route donor chunks for the identified deficit;
3. make a controlled number of updates;
4. evaluate competence/return;
5. restore checkpoint;
6. compare with:
   - random experience;
   - already-mastered experience;
   - globally high-TD-error experience.

This is the key diagnostic before full online routing.

## 19.4. Experiment 4 — Routing comparison

Compare:

```text
B1 Independent
B2 Shared Replay
B3 Random Routing
B4 SUPER-style TD Priority
B5 QMP-style
B6 Greedy
B7 UOT
```

Primary metrics:

- average episodic return;
- success rate;
- area under learning curve;
- environment interactions to threshold;
- mean population performance;
- worst-policy performance;
- positive-transfer rate;
- negative-transfer rate;
- improvement per routed transition;
- routing overhead.

## 19.5. Experiment 5 — Core ablations

### A1. No dynamic experiences

Use fixed-length windows only.

Question:

```text
Does functional grouping matter?
```

### A2. No minimax deficit

Route from strongest donors uniformly/randomly.

Question:

```text
Does receiver-need estimation matter?
```

### A3. Greedy instead of OT

Question:

```text
Does transport-level assignment matter?
```

### A4. No success-support filter

Keep all experience clusters.

Question:

```text
Does filtering repeated successful experience matter?
```

### A5. Population size

```text
N ∈ {2, 4, 8}
```

Only after `N=4` works.

### A6. Routing budget

Vary number/fraction of routed transitions.

Question:

```text
Does performance improve because of routing quality or simply because of more shared data?
```

---

# 20. Fair Budget Accounting

This is mandatory.

For all population methods:

```python
total_environment_interactions =
    sum_i interactions_by_policy_i
```

Primary population comparisons must match:

```text
same N
same total interactions
same policy architecture
same update count
same local replay batch size
same routing bandwidth where applicable
```

For `Single SAC`, report:

### Primary

```text
single policy gets the same total interaction budget
```

This tests population/sample-efficiency fairly.

### Supplementary

```text
single policy gets the same per-policy interaction budget
```

This shows the practical benefit/cost of parallel population training.

Always report coordinator wall-clock overhead separately.

---

# 21. Logging Requirements

Use WandB or an equivalent logger.

Every run should log:

## RL

```text
env_steps_total
env_steps_per_policy
return/policy_i
success/policy_i
critic_loss/policy_i
actor_loss/policy_i
```

## Experience discovery

```text
num_experiences
num_valid_experiences
chunks_per_episode
chunk_length_mean
cluster_size_histogram
new_experiences_per_refresh
experience_merge_count
```

## Competence

```text
competence_mean[i,e]
competence_trials[i,e]
frontier[e]
deficit[i,e]
```

## Routing

```text
route_count
route_mass
donor_distribution
receiver_distribution
experience_distribution
routed_batch_fraction
positive_transfer_rate
negative_transfer_rate
```

## Compute

```text
wall_clock
routing_time
OT_solver_time
policy_update_time
```

---

# 22. Unit Tests

## Segmenter

- boundaries sorted;
- no overlapping chunks;
- all timesteps covered exactly once;
- minimum chunk length respected.

## Grouper

- identical precondition/effect → same experience;
- large precondition/effect difference → different experience;
- prototype update deterministic.

## Validity

- repeated successful cluster becomes valid;
- rare cluster remains provisional.

## Competence

Synthetic test:

```text
policy A succeeds 8/10 opportunities
policy B succeeds 2/10
```

Expected:

```text
C[A,e] > C[B,e]
```

## Deficit

Expected:

```text
frontier = max competence
best policy deficit = 0
weaker policy deficit > 0
```

## OT

Synthetic matrix where:

```text
A strong on e1
B weak on e1
B strong on e2
A weak on e2
```

Expected transport:

```text
A:e1 → B:e1
B:e2 → A:e2
```

This unit test should pass before any real routing experiment.

---

# 23. Configuration Defaults

Initial values only:

```yaml
population:
  size: 4

experience:
  K_seg: 6
  median_window: 7
  min_chunk_len: 4
  eps_experience: TBD_from_calibration
  eps_merge: 0.7 * eps_experience
  min_success_support: 5
  success_support_threshold: 0.05
  refresh_interval: 10000

competence:
  beta_alpha: 1.0
  beta_beta: 1.0
  min_opportunities: 5
  horizon: task_dependent

routing:
  interval: 5000
  budget_chunks: 32
  route_batch_fraction: 0.25

ot:
  solver: unbalanced_sinkhorn
  reg: 0.05
  reg_m: 1.0
  big_cost: 1000.0

evaluation:
  eval_interval: 10000
  episodes_per_policy: 20
```

Do not lock thresholds before plotting empirical distance distributions.

---

# 24. Calibration Procedure for Experience Thresholds

Do not choose `eps_experience` arbitrarily.

On a pilot dataset:

1. sample `10k–50k` chunks;
2. compute nearest-neighbor state-transition descriptor distances;
3. plot distance histogram;
4. inspect nearest neighbors manually;
5. choose an initial threshold around the transition between clearly same-transition and clearly different-transition pairs;
6. sweep 3 values in the ablation.

Record:

```text
eps_experience
→ vocabulary size
→ average cluster size
→ cluster purity by manual inspection
→ final routing performance
```

---

# 25. MVP Stop/Go Criteria

## Gate A — RL backbone

GO if:

```text
single SAC reliably learns the first task
```

Otherwise fix RL first.

## Gate B — Experience discovery

GO if:

- vocabulary does not explode;
- clusters have repeated support;
- representative chunks show recognizable common state transitions.

Otherwise adjust state representation / thresholds, not the routing algorithm.

## Gate C — Complementary competence

GO if:

```text
different policies are strongest on a non-trivial set of experiences
```

If one policy dominates all cells, increase controlled exploration/environment diversity or reconsider the population hypothesis.

## Gate D — Greedy transfer

GO to OT only if greedy deficit-based routing improves over:

```text
independent
or
random/share-all
```

If not, debug the competence signal and routed replay.

## Gate E — OT

Keep OT in the final method only if it improves over greedy assignment or gives a meaningful robustness/budget advantage.

If greedy is just as good, remove OT rather than forcing it into the paper.

---

# 26. What Not to Optimize Prematurely

Do not spend time on:

- perfect semantic skill names;
- trajectory-language alignment;
- VLM embeddings;
- image encoders;
- learned router architecture;
- fancy OT costs;
- policy-dynamics divergence estimator;
- large populations;
- many Meta-World tasks;
- real-world robot data;
- large seed sweeps.

First answer:

```text
Does the simple pipeline produce a useful non-trivial routing signal?
```

---

# 27. First Coding Order for an Agent

The coding agent should execute in exactly this order:

1. set up current Meta-World;
2. implement/port clean SAC;
3. train one policy;
4. generalize to `N` independent policies;
5. implement shared replay + random sharing;
6. save complete trajectories;
7. implement EXTRACT-style state-transition segmenter;
8. implement chunk visualization;
9. implement state-transition experience grouping;
10. implement success-support validity filtering;
11. implement opportunity-based competence estimation;
12. implement deficit matrix;
13. implement greedy routing;
14. validate controlled transfer;
15. integrate POT;
16. implement UOT router;
17. implement SUPER-style TD priority;
18. implement QMP-style receiver-Q baseline;
19. run B0–B7;
20. only then consider second environment / stronger chunking.

---

# 28. Definition of Done for the MVP

The MVP is complete when one command can run:

```bash
python scripts/run_full.py \
    --env <metaworld-task> \
    --population-size 4 \
    --router uot
```

and automatically produce:

1. learning curves for every policy;
2. population mean/worst/best performance;
3. discovered experience vocabulary over time;
4. representative chunks per experience;
5. competence heatmap;
6. deficit heatmap;
7. UOT routing matrix;
8. routing counts and budgets;
9. comparison with:
   - independent;
   - shared replay;
   - random routing;
   - TD-priority sharing;
   - QMP-style;
   - greedy routing;
10. matched interaction-budget summary.

---

# 29. Repository/Baseline Decision Summary

| Prior work / repo | Use as codebase? | Use as experiment baseline? | Exact role |
|---|---:|---:|---|
| `clvrai/qmp` | **Yes, primary reference** | **Yes** | Multi-policy off-policy organization; QMP-style receiver-Q behavior sharing |
| `jesbu1/extract` | Reference only | Optional ablation | EXTRACT-style cluster → smooth → chunk pipeline |
| `mgerstgrasser/super` | No | **Yes, adapted** | Global high-TD-error selective sharing |
| `uoe-agents/seac` | No | Optional | Shared cross-agent experience / importance weighting |
| `Farama-Foundation/Metaworld` | **Yes** | Environment | Primary MVP simulation suite |
| `mani-skill/ManiSkill` | Later | Later | Second-suite validation |
| `PythonOT/POT` | **Yes, dependency** | N/A | UOT / partial OT solver |

---

# 30. Scientific Claim the MVP Is Designed to Test

Do not claim:

```text
"We solve semantic skill discovery."
"We introduce multi-agent experience sharing."
"We introduce population RL."
"We introduce optimal transport for RL."
```

The MVP is designed to test the narrower claim:

> A population can be organized into a dynamic functional-experience competence space; the resulting receiver deficits can be used to route experience from policies that have already mastered a functional transition to policies that have not, outperforming indiscriminate or globally prioritized experience sharing.

The core ablation hierarchy should therefore be:

```text
Independent
    ↓
Shared / global-priority experience
    ↓
Receiver deficit + greedy donor matching
    ↓
Receiver deficit + UOT assignment
```

If this hierarchy is empirically supported, then the project has a defensible foundation for the full paper.
